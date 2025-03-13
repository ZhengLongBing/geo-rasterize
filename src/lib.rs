#![doc = include_str!("../README.md")]
use std::{collections::HashSet, fmt::Debug, ops::Add};

use euclid::{Transform2D, UnknownUnit};
use geo::{
    algorithm::{
        coords_iter::CoordsIter,
        map_coords::{MapCoords, MapCoordsInPlace},
    },
    Coord, Geometry, GeometryCollection, Line, LineString, MultiLineString, MultiPoint,
    MultiPolygon, Point, Polygon, Rect, Triangle,
};
use ndarray::s;
use ndarray::Array2;
use num_traits::{Num, NumCast};
use thiserror::Error;

mod poly;
use poly::rasterize_polygon;
mod line;
use line::rasterize_line;
#[cfg(test)]
mod proptests;

/// 仿射变换，用于描述如何将世界空间坐标转换为像素坐标。
/// 此变换允许几何形状从其原始坐标系统映射到栅格图像的像素网格上。
pub type Transform = Transform2D<f64, UnknownUnit, UnknownUnit>;
/// 欧几里得点类型，用于内部坐标计算和转换操作。
type EuclidPoint = euclid::Point2D<f64, UnknownUnit>;

/// 本库的错误类型
/// 包含栅格化过程中可能遇到的各种错误
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum RasterizeError {
    /// 提供的几何图形中至少有一个坐标是NaN或无限的
    /// 这会导致无法准确进行栅格化计算
    #[error("提供的几何图形中至少有一个坐标是NaN或无限的")]
    NonFiniteCoordinate,

    /// 构建器中缺少必需的`width`参数
    /// 栅格化需要明确的宽度值以确定输出数组的大小
    #[error("构建器中缺少必需的`width`参数")]
    MissingWidth,

    /// 构建器中缺少必需的`height`参数
    /// 栅格化需要明确的高度值以确定输出数组的大小
    #[error("构建器中缺少必需的`height`参数")]
    MissingHeight,
}

/// 本库使用的结果类型，使用[RasterizeError]作为错误类型。
/// 这为库的所有操作提供了一致的错误处理方式。
pub type Result<T> = std::result::Result<T, RasterizeError>;

/// 二值栅格化器构建器，用于构造[BinaryRasterizer]实例。
///
/// [BinaryRasterizer]是一个能够将几何形状栅格化为二维布尔数组的工具。
/// 使用此构建器可以方便地配置栅格化参数，如宽度、高度和坐标变换。
///
/// # 示例
///
/// ```rust
/// # use geo_rasterize::{Result, BinaryBuilder, BinaryRasterizer};
/// # fn main() -> Result<()> {
/// let rasterizer: BinaryRasterizer = BinaryBuilder::new().width(37).height(21).build()?;
/// # Ok(())}
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryBuilder {
    /// 输出栅格图像的宽度（像素数）
    width: Option<usize>,
    /// 输出栅格图像的高度（像素数）
    height: Option<usize>,
    /// 可选的从地理坐标到像素坐标的变换矩阵
    geo_to_pix: Option<Transform>,
}

impl BinaryBuilder {
    /// 创建一个新的二值栅格化器构建器实例
    ///
    /// 返回一个默认配置的构建器，需要通过链式调用进一步配置
    pub fn new() -> Self {
        BinaryBuilder::default()
    }

    /// 设置输出栅格图像的宽度
    ///
    /// # 参数
    /// * `width` - 栅格图像的宽度，以像素为单位
    ///
    /// # 返回
    /// 返回修改后的构建器实例，支持链式调用
    pub fn width(mut self, width: usize) -> Self {
        self.width = Some(width);
        self
    }

    /// 设置输出栅格图像的高度
    ///
    /// # 参数
    /// * `height` - 栅格图像的高度，以像素为单位
    ///
    /// # 返回
    /// 返回修改后的构建器实例，支持链式调用
    pub fn height(mut self, height: usize) -> Self {
        self.height = Some(height);
        self
    }

    /// 设置从地理坐标到像素坐标的变换矩阵
    ///
    /// # 参数
    /// * `geo_to_pix` - 仿射变换矩阵，用于将世界坐标转换为像素坐标
    ///
    /// # 返回
    /// 返回修改后的构建器实例，支持链式调用
    pub fn geo_to_pix(mut self, geo_to_pix: Transform) -> Self {
        self.geo_to_pix = Some(geo_to_pix);
        self
    }

    /// 根据当前配置构建一个二值栅格化器实例
    ///
    /// # 返回
    /// 成功时返回配置好的二值栅格化器；失败时返回相应的错误
    ///
    /// # 错误
    /// 如果缺少必需的宽度或高度参数，将返回相应的错误
    pub fn build(self) -> Result<BinaryRasterizer> {
        match (self.width, self.height) {
            (None, _) => Err(RasterizeError::MissingWidth),
            (_, None) => Err(RasterizeError::MissingHeight),
            (Some(width), Some(height)) => BinaryRasterizer::new(width, height, self.geo_to_pix),
        }
    }
}

/// 二值栅格化器，将几何形状栅格化为二维布尔数组。
///
/// 该栅格化器可以通过调用[BinaryRasterizer::new]或使用[BinaryBuilder]来构建。
///
/// 每个栅格化器需要指定以像素为单位的`width`和`height`，用于描述输出数组的形状。
/// 可以选择提供一个仿射变换，用于将世界空间坐标转换为像素空间坐标。
/// 当提供变换矩阵时，其所有参数必须是有限的，否则将返回[RasterizeError::NonFiniteCoordinate]。
///
/// # 示例
///
/// ```rust
/// # use geo_rasterize::{Result, BinaryBuilder, BinaryRasterizer};
/// # fn main() -> Result<()> {
/// use geo::{Geometry, Line, Point};
/// use ndarray::array;
/// use geo_rasterize::BinaryBuilder;
///
/// // 创建几何形状集合：一个点和一条线
/// let shapes: Vec<Geometry<i32>> =
///     vec![Point::new(3, 4).into(),
///          Line::new((0, 3), (3, 0)).into()];
///
/// // 构建一个4x5像素的栅格化器
/// let mut r = BinaryBuilder::new().width(4).height(5).build()?;
///
/// // 栅格化每个形状
/// for shape in shapes {
///     r.rasterize(&shape)?;
/// }
///
/// // 获取结果像素数组
/// let pixels = r.finish();
/// assert_eq!(
///     pixels.mapv(|v| v as u8),
///     array![
///         [0, 0, 1, 0],
///         [0, 1, 1, 0],
///         [1, 1, 0, 0],
///         [1, 0, 0, 0],
///         [0, 0, 0, 1]
///     ]
/// );
/// # Ok(())}
/// ```
#[derive(Clone, Debug)]
pub struct BinaryRasterizer {
    /// 内部使用的通用栅格化器，使用u8类型作为标签
    inner: Rasterizer<u8>,
}

fn to_float<T>(coords: Coord<T>) -> Coord<f64>
where
    T: Into<f64> + Copy + NumCast + Debug + PartialOrd + Num, // T必须可复制且可转换为f64
{
    Coord {
        x: coords.x.into(),
        y: coords.y.into(),
    }
}
impl BinaryRasterizer {
    /// 创建一个新的二值栅格化器实例
    ///
    /// # 参数
    /// * `width` - 输出栅格图像的宽度（像素单位）
    /// * `height` - 输出栅格图像的高度（像素单位）
    /// * `geo_to_pix` - 可选的坐标变换矩阵，用于将几何坐标转换为像素坐标
    ///
    /// # 返回值
    /// 返回Result类型，成功时包含BinaryRasterizer实例，失败时包含错误信息
    ///
    /// # 错误
    /// 如果提供的变换矩阵中包含非有限值（NaN或无限），则返回NonFiniteCoordinate错误
    pub fn new(width: usize, height: usize, geo_to_pix: Option<Transform>) -> Result<Self> {
        // 检查变换矩阵是否包含非有限值
        let non_finite = geo_to_pix
            .map(|geo_to_pix| geo_to_pix.to_array().iter().any(|param| !param.is_finite()))
            .unwrap_or(false);

        if non_finite {
            // 存在非有限值，返回错误
            Err(RasterizeError::NonFiniteCoordinate)
        } else {
            // 创建内部通用栅格化器，使用替换策略和背景值0
            let inner = Rasterizer::new(width, height, geo_to_pix, MergeAlgorithm::Replace, 0);
            Ok(BinaryRasterizer { inner })
        }
    }

    /// 获取当前使用的坐标变换矩阵
    ///
    /// # 返回值
    /// 返回可选的变换矩阵，如果未设置则返回None
    pub fn geo_to_pix(&self) -> Option<Transform> {
        self.inner.geo_to_pix
    }

    /// 栅格化单个几何形状
    ///
    /// 该方法可以处理[geo]库提供的任何几何类型，支持任何可转换为f64的坐标数值类型
    ///
    /// # 类型参数
    /// * `Coord` - 输入几何形状的坐标类型
    /// * `InputShape` - 输入几何形状类型
    /// * `ShapeAsF64` - 坐标转换为f64后的几何形状类型
    ///
    /// # 参数
    /// * `shape` - 要栅格化的几何形状引用
    ///
    /// # 返回值
    /// 成功时返回()，失败时返回错误信息
    ///
    /// # 泛型约束
    /// * InputShape必须能够将坐标从Coord类型映射到f64类型
    /// * ShapeAsF64必须实现Rasterize、CoordsIter和MapCoordsInplace特性
    /// * Coord必须可转换为f64，可复制，可调试，是数值类型，可进行数值类型转换，支持偏序比较
    pub fn rasterize<Coord, InputShape, ShapeAsF64>(&mut self, shape: &InputShape) -> Result<()>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize<u8> + for<'a> CoordsIter<Scalar = f64> + MapCoordsInPlace<f64>,
        Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
    {
        // 调用内部栅格化器的栅格化方法，使用标签值1表示形状存在的像素
        self.inner.rasterize(shape, 1)
    }

    /// 完成栅格化并获取结果二值数组
    ///
    /// # 返回值
    /// 返回一个二维布尔数组，其中true表示形状存在的像素，false表示背景
    pub fn finish(self) -> Array2<bool> {
        // 将内部的u8数组转换为布尔数组，值为1的像素转换为true，其他值转换为false
        self.inner
            .finish()
            .mapv(|v| if v == 1u8 { true } else { false })
    }
}

/// 定义栅格化能力的特性
///
/// 该特性用于表示一个类型可以被栅格化到一个二维数组中
/// 不会在公开API中直接使用，主要用于内部实现
///
/// # 类型参数
/// * `Label` - 栅格化后像素的标签类型，必须支持复制和加法操作
#[doc(hidden)]
pub trait Rasterize<Label>
where
    Label: Copy + Add<Output = Label>,
{
    /// 将自身栅格化到指定的栅格化器中
    ///
    /// # 参数
    /// * `rasterizer` - 用于执行栅格化操作的栅格化器实例
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>);
}

/// 像素冲突解决策略
///
/// 当两个几何形状覆盖同一个像素时，使用该枚举决定如何处理重叠部分
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeAlgorithm {
    /// 替换策略：使用最后绘制的形状的标签值覆盖像素
    /// 这是默认策略
    Replace,

    /// 累加策略：将所有覆盖该像素的形状的标签值相加
    /// 适用于需要计算重叠次数或强度的场景
    Add,
}

impl Default for MergeAlgorithm {
    /// 设置默认的合并算法为替换策略
    fn default() -> Self {
        MergeAlgorithm::Replace
    }
}

/// 通用栅格化器构建器，用于构造[Rasterizer]实例
///
/// 与[BinaryRasterizer]产生布尔数组不同，[Rasterizer]可以产生任意标签类型的数组，
/// 该标签类型必须实现[Copy][std::marker::Copy]和[Add][std::ops::Add]特性，
/// 通常使用数值类型作为标签。
///
/// # 类型参数
/// * `Label` - 栅格化后像素的标签类型
///
/// # 说明
/// [LabelBuilder]需要指定`Label`类型，因此创建构建器的唯一方式是指定一个背景值。
/// 背景值(`background`)用于初始化栅格数组，对应于"无形状"的像素值。
/// 通常，您会使用`LabelBuilder::background(0)`或`LabelBuilder::background(0f32)`开始构建。
///
/// 除了`background`、`width`和`height`外，您还可以提供一个[MergeAlgorithm]来指定
/// 当两个不同的形状填充同一个像素时栅格化器应该如何处理。如果不提供任何算法，
/// 栅格化器将默认使用[Replace][MergeAlgorithm::Replace]策略。
///
/// # 示例
///
/// ```rust
/// # fn main() -> geo_rasterize::Result<()> {
/// use geo_rasterize::LabelBuilder;
///
/// let mut rasterizer = LabelBuilder::background(0i32).width(4).height(5).build()?;
/// # Ok(())}
/// ```
#[derive(Debug, Clone, Default)]
pub struct LabelBuilder<Label> {
    /// 背景值，用于初始化栅格数组
    background: Label,
    /// 输出栅格图像的宽度（像素）
    width: Option<usize>,
    /// 输出栅格图像的高度（像素）
    height: Option<usize>,
    /// 可选的几何坐标到像素坐标的变换矩阵
    geo_to_pix: Option<Transform>,
    /// 可选的像素合并算法
    algorithm: Option<MergeAlgorithm>,
}
/// 标签构建器的实现
impl<Label> LabelBuilder<Label>
where
    Label: Copy + Add<Output = Label>, // Label类型必须可复制且支持加法运算
{
    /// 创建一个新的标签构建器实例，设置背景值
    ///
    /// # 参数
    /// * `background` - 用于初始化栅格数组的背景值
    ///
    /// # 返回值
    /// 返回配置了背景值的构建器实例
    pub fn background(background: Label) -> Self {
        LabelBuilder {
            background,
            width: None,
            height: None,
            geo_to_pix: None,
            algorithm: None,
        }
    }

    /// 设置栅格图像的宽度
    ///
    /// # 参数
    /// * `width` - 栅格图像宽度（像素）
    pub fn width(mut self, width: usize) -> Self {
        self.width = Some(width);
        self
    }

    /// 设置栅格图像的高度
    ///
    /// # 参数
    /// * `height` - 栅格图像高度（像素）
    pub fn height(mut self, height: usize) -> Self {
        self.height = Some(height);
        self
    }

    /// 设置几何坐标到像素坐标的变换矩阵
    ///
    /// # 参数
    /// * `geo_to_pix` - 坐标变换矩阵
    pub fn geo_to_pix(mut self, geo_to_pix: Transform) -> Self {
        self.geo_to_pix = Some(geo_to_pix);
        self
    }

    /// 设置像素合并算法
    ///
    /// # 参数
    /// * `algorithm` - 用于处理重叠像素的合并算法
    pub fn algorithm(mut self, algorithm: MergeAlgorithm) -> Self {
        self.algorithm = Some(algorithm);
        self
    }

    /// 构建栅格化器实例
    ///
    /// # 返回值
    /// 返回Result类型，成功时包含配置好的Rasterizer实例，
    /// 失败时返回相应的错误（缺少宽度或高度）
    pub fn build(self) -> Result<Rasterizer<Label>> {
        match (self.width, self.height) {
            (None, _) => Err(RasterizeError::MissingWidth),
            (_, None) => Err(RasterizeError::MissingHeight),
            (Some(width), Some(height)) => Ok(Rasterizer::new(
                width,
                height,
                self.geo_to_pix,
                self.algorithm.unwrap_or_default(),
                self.background,
            )),
        }
    }
}

/// 通用栅格化器，可以将几何形状栅格化为任意标签类型的二维数组
///
/// 与[BinaryRasterizer]不同，该栅格化器可以生成任意实现了[Copy]和[Add]特征的
/// 标签类型数组，通常使用数值类型作为标签。
///
/// 可以通过调用[Rasterizer::new]或使用[LabelBuilder]来构建实例。构建时需要
/// 指定：
/// - `width`：输出图像宽度
/// - `height`：输出图像高度
/// - `background`：背景像素的默认值
/// - `geo_to_pix`：可选的坐标变换矩阵
/// - `algorithm`：可选的像素合并算法，默认使用[MergeAlgorithm::Replace]
///
/// # 示例
///
/// ```rust
/// # use geo_rasterize::{Result, LabelBuilder, Rasterizer};
/// # fn main() -> Result<()> {
/// use geo::{Geometry, Line, Point};
/// use ndarray::array;
///
/// let point = Point::new(3, 4);
/// let line = Line::new((0, 3), (3, 0));
///
/// let mut rasterizer = LabelBuilder::background(0).width(4).height(5).build()?;
/// rasterizer.rasterize(&point, 7)?;
/// rasterizer.rasterize(&line, 3)?;
///
/// let pixels = rasterizer.finish();
/// assert_eq!(
///     pixels.mapv(|v| v as u8),
///     array![
///         [0, 0, 3, 0],
///         [0, 3, 3, 0],
///         [3, 3, 0, 0],
///         [3, 0, 0, 0],
///         [0, 0, 0, 7]
///     ]
/// );
/// # Ok(())}
/// ```
#[derive(Clone, Debug)]
pub struct Rasterizer<Label> {
    /// 存储栅格化结果的二维数组
    pixels: Array2<Label>,
    /// 可选的坐标变换矩阵
    geo_to_pix: Option<Transform>,
    /// 像素合并算法
    algorithm: MergeAlgorithm,
    /// 当前使用的前景值
    foreground: Label,
    /// 上一次栅格化操作中被填充的像素坐标集合
    previous_burnt_points: HashSet<(usize, usize)>,
    /// 当前栅格化操作中被填充的像素坐标集合
    current_burnt_points: HashSet<(usize, usize)>,
}
/// 通用栅格化器的实现
impl<Label> Rasterizer<Label>
where
    Label: Copy + Add<Output = Label>, // Label类型必须实现Copy和Add特征
{
    /// 创建一个新的栅格化器实例
    ///
    /// # 参数
    /// * `width` - 输出栅格图像的宽度（像素）
    /// * `height` - 输出栅格图像的高度（像素）
    /// * `geo_to_pix` - 可选的坐标变换矩阵，用于将几何坐标转换为像素坐标
    /// * `algorithm` - 像素合并算法
    /// * `background` - 背景像素值
    pub fn new(
        width: usize,
        height: usize,
        geo_to_pix: Option<Transform>,
        algorithm: MergeAlgorithm,
        background: Label,
    ) -> Self {
        let pixels = Array2::from_elem((height, width), background);
        Rasterizer {
            pixels,
            geo_to_pix,
            algorithm,
            foreground: background,
            previous_burnt_points: HashSet::new(),
            current_burnt_points: HashSet::new(),
        }
    }

    /// 获取栅格图像的宽度
    fn width(&self) -> usize {
        self.pixels.shape()[1]
    }

    /// 获取栅格图像的高度
    fn height(&self) -> usize {
        self.pixels.shape()[0]
    }

    /// 获取当前使用的坐标变换矩阵
    pub fn geo_to_pix(&self) -> Option<Transform> {
        self.geo_to_pix
    }

    /// 开始处理新的线串时清空所有已填充点集
    ///
    /// 在MergeAlgorithm::Add模式下，需要确保不会重复填充同一个像素。
    /// 我们维护两个索引集合:
    /// - previous_burnt_points: 记录当前线串中上一条线段已填充的像素
    /// - current_burnt_points: 记录当前线段已填充的像素
    fn new_linestring(&mut self) {
        self.previous_burnt_points.clear();
        self.current_burnt_points.clear();
    }

    /// 开始处理新的线段时交换并清空当前点集
    fn new_line(&mut self) {
        std::mem::swap(
            &mut self.previous_burnt_points,
            &mut self.current_burnt_points,
        );
        self.current_burnt_points.clear();
    }

    /// 填充单个像素
    ///
    /// # 参数
    /// * `ix` - 像素x坐标
    /// * `iy` - 像素y坐标
    fn fill_pixel(&mut self, ix: usize, iy: usize) {
        debug_assert!(ix < self.width());
        debug_assert!(iy < self.height());
        let mut slice = self.pixels.slice_mut(s![iy, ix]);
        match self.algorithm {
            MergeAlgorithm::Replace => slice.fill(self.foreground),
            MergeAlgorithm::Add => {
                slice.mapv_inplace(|v| v + self.foreground);
            }
        }
    }

    /// 填充单个像素，但避免重复填充
    ///
    /// # 参数
    /// * `ix` - 像素x坐标
    /// * `iy` - 像素y坐标
    /// * `use_current_too` - 是否同时检查当前点集
    fn fill_pixel_no_repeat(&mut self, ix: usize, iy: usize, use_current_too: bool) {
        match self.algorithm {
            MergeAlgorithm::Replace => {
                self.fill_pixel(ix, iy);
            }
            MergeAlgorithm::Add => {
                let point = (ix, iy);
                let mut do_fill_pixel = !self.previous_burnt_points.contains(&point);
                if use_current_too {
                    do_fill_pixel = do_fill_pixel && !self.current_burnt_points.contains(&point);
                }
                if do_fill_pixel {
                    self.fill_pixel(ix, iy);
                    self.current_burnt_points.insert(point);
                }
            }
        }
    }

    /// 填充水平线段
    ///
    /// 栅格化算法对写入顺序非常敏感。由于主要处理水平线，
    /// 当水平相邻的像素在内存中也相邻时(即数组最后一维是x)性能最佳。
    ///
    /// # 参数
    /// * `x_start` - 起始x坐标(不包含)
    /// * `x_end` - 结束x坐标(不包含)
    /// * `y` - y坐标
    fn fill_horizontal_line(&mut self, x_start: usize, x_end: usize, y: usize) {
        let mut slice = self.pixels.slice_mut(s![y, x_start..x_end]);
        match self.algorithm {
            MergeAlgorithm::Replace => slice.fill(self.foreground),
            MergeAlgorithm::Add => {
                slice.mapv_inplace(|v| v + self.foreground);
            }
        }
    }

    /// 填充水平线段，但避免重复填充
    fn fill_horizontal_line_no_repeat(&mut self, x_start: usize, x_end: usize, y: usize) {
        for x in x_start..=x_end {
            self.fill_pixel_no_repeat(x, y, true);
        }
    }

    /// 填充垂直线段，但避免重复填充
    fn fill_vertical_line_no_repeat(&mut self, x: usize, y_start: usize, y_end: usize) {
        for y in y_start..=y_end {
            self.fill_pixel_no_repeat(x, y, false);
        }
    }

    /// 栅格化一个几何形状
    ///
    /// # 参数
    /// * `shape` - 输入的几何形状，可以是geo库提供的任何类型
    /// * `foreground` - 前景像素值
    ///
    /// # 类型参数
    /// * `Coord` - 输入坐标类型，必须可以转换为f64
    /// * `InputShape` - 输入形状类型
    /// * `ShapeAsF64` - 转换为f64坐标后的形状类型
    ///
    /// # 错误
    /// 如果坐标包含非有限值，返回NonFiniteCoordinate错误
    pub fn rasterize<Coord, InputShape, ShapeAsF64>(
        &mut self,
        shape: &InputShape,
        foreground: Label,
    ) -> Result<()>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize<Label> + for<'a> CoordsIter<Scalar = f64> + MapCoordsInPlace<f64>,
        Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
    {
        // 将输入坐标转换为f64类型
        let mut float = shape.map_coords(to_float);

        // 确保所有坐标都是有限值
        let all_finite = float
            .coords_iter()
            .all(|coordinate| coordinate.x.is_finite() && coordinate.y.is_finite());
        if !all_finite {
            return Err(RasterizeError::NonFiniteCoordinate);
        }

        self.foreground = foreground;

        // 如果提供了变换矩阵，将地理坐标转换为像素坐标
        match self.geo_to_pix {
            None => float,
            Some(transform) => {
                float.map_coords_in_place(|coord| {
                    transform
                        .transform_point(EuclidPoint::new(coord.x, coord.y))
                        .to_tuple()
                        .into()
                });
                float
            }
        }
        .rasterize(self);

        Ok(())
    }

    /// 获取完成的栅格数组
    pub fn finish(self) -> Array2<Label> {
        self.pixels
    }
}
/// 为Point<f64>类型实现栅格化特征
///
/// # 类型参数
/// * `Label` - 栅格化后的标签类型，必须实现Copy和Add特征
impl<Label> Rasterize<Label> for Point<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 将点栅格化为单个像素
    ///
    /// 只有当点的坐标为非负且在栅格范围内时才进行栅格化
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        // 检查坐标是否为非负
        if self.x() >= 0. && self.y() >= 0. {
            // 将浮点坐标向下取整转换为像素索引
            let x = self.x().floor() as usize;
            let y = self.y().floor() as usize;
            // 检查像素是否在栅格范围内
            if x < rasterizer.width() && y < rasterizer.height() {
                rasterizer.fill_pixel(x, y);
            }
        }
    }
}

/// 为MultiPoint<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for MultiPoint<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 栅格化多点集合中的每个点
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.iter().for_each(|point| point.rasterize(rasterizer));
    }
}

/// 为Rect<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for Rect<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 将矩形转换为多边形进行栅格化
    ///
    /// 虽然直接实现矩形栅格化可能更快，但考虑到仿射变换可能会旋转或剪切矩形，
    /// 使其不再与坐标轴对齐，这里选择将其转换为多边形处理
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.to_polygon().rasterize(rasterizer);
    }
}

/// 为Line<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for Line<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 栅格化单条线段
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        rasterizer.new_linestring();
        rasterize_line(self, rasterizer);
    }
}

/// 为LineString<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for LineString<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 栅格化线串中的每个线段
    ///
    /// 注意：虽然可以将闭合的LineString视为无孔多边形并填充，
    /// 但为了与GDAL保持一致，这里仍将其作为线段序列处理。
    /// GDAL中的LinearRings被视为LineSegments而不会被填充。
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        rasterizer.new_linestring();
        self.lines().for_each(|line| {
            rasterizer.new_line();
            rasterize_line(&line, rasterizer);
        });
    }
}

/// 为MultiLineString<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for MultiLineString<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 栅格化多线串中的每条线串
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.iter()
            .for_each(|line_string| line_string.rasterize(rasterizer));
    }
}

/// 为Polygon<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for Polygon<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 栅格化多边形，包括外边界和内部孔洞
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        rasterize_polygon(self.exterior(), self.interiors(), rasterizer);
    }
}

/// 为MultiPolygon<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for MultiPolygon<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 栅格化多多边形中的每个多边形
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.iter().for_each(|poly| poly.rasterize(rasterizer));
    }
}

/// 为Triangle<f64>类型实现栅格化特征
impl<Label> Rasterize<Label> for Triangle<f64>
where
    Label: Copy + Add<Output = Label>,
{
    /// 将三角形转换为多边形进行栅格化
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.to_polygon().rasterize(rasterizer)
    }
}

/// 为Geometry<f64>类型实现栅格化特征
///
/// # 类型参数
/// * `Label` - 栅格化时使用的标签类型,需要实现Copy和Add特征
///
/// # 说明
/// 这个实现允许将任何几何体类型转换为栅格表示。根据几何体的具体类型,
/// 调用相应的栅格化实现:
/// - 点(Point)
/// - 线(Line)
/// - 线串(LineString)
/// - 多边形(Polygon)
/// - 几何集合(GeometryCollection)
/// - 多点(MultiPoint)
/// - 多线串(MultiLineString)
/// - 多多边形(MultiPolygon)
/// - 矩形(Rect)
/// - 三角形(Triangle)
impl<Label> Rasterize<Label> for Geometry<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        match self {
            // 对每种几何类型调用其对应的栅格化实现
            Geometry::Point(point) => point.rasterize(rasterizer),
            Geometry::Line(line) => line.rasterize(rasterizer),
            Geometry::LineString(ls) => ls.rasterize(rasterizer),
            Geometry::Polygon(poly) => poly.rasterize(rasterizer),
            Geometry::GeometryCollection(gc) => gc.rasterize(rasterizer),
            Geometry::MultiPoint(points) => points.rasterize(rasterizer),
            Geometry::MultiLineString(lines) => lines.rasterize(rasterizer),
            Geometry::MultiPolygon(polys) => polys.rasterize(rasterizer),
            Geometry::Rect(rect) => rect.rasterize(rasterizer),
            Geometry::Triangle(tri) => tri.rasterize(rasterizer),
        }
    }
}

/// 为GeometryCollection<f64>类型实现栅格化特征
///
/// # 类型参数
/// * `Label` - 栅格化时使用的标签类型,需要实现Copy和Add特征
///
/// # 说明
/// 这个实现允许将几何集合转换为栅格表示。它会遍历集合中的每个几何体,
/// 并对每个几何体调用其栅格化实现。
impl<Label> Rasterize<Label> for GeometryCollection<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        // 遍历集合中的每个几何体并进行栅格化
        self.iter().for_each(|thing| thing.rasterize(rasterizer));
    }
}

#[cfg(test)]
mod tests;
