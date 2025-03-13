use std::{iter::once, ops::Add};

use geo::{coords_iter::CoordsIter, winding_order::Winding, LineString, Point};
use itertools::Itertools;

use crate::{Rasterize, Rasterizer};

/// 获取多边形所有顶点的y坐标
///
/// # 参数
/// * `first` - 多边形外环线串
/// * `rest` - 多边形内环线串数组
///
/// # 返回值
/// 返回包含所有顶点y坐标的迭代器
fn y_coordinates<'a>(
    first: &'a LineString<f64>,
    rest: &'a [LineString<f64>],
) -> impl Iterator<Item = isize> + 'a {
    once(first)
        .chain(rest)
        .flat_map(|line_string| line_string.points().map(|point| point.y().floor() as isize))
}

/// 定义点对类型,表示线段的两个端点
type PointPair = (Point<f64>, Point<f64>);

/// 将多边形的所有点按顺时针顺序转换为点对序列
///
/// # 参数
/// * `first` - 多边形外环线串
/// * `rest` - 多边形内环线串数组
///
/// # 返回值
/// 返回包含所有相邻点对的向量
fn into_pointpairs(first: &LineString<f64>, rest: &[LineString<f64>]) -> Vec<PointPair> {
    // 预分配向量容量以提高性能
    let num_pairs = once(first).chain(rest).map(|ls| ls.0.len() - 1).sum();
    let mut result = Vec::with_capacity(num_pairs);

    // 遍历所有线串,保持顺时针方向
    for ls in once(first).chain(rest) {
        if ls.is_cw() {
            result.extend(ls.points().tuple_windows::<PointPair>());
        } else {
            result.extend(ls.points().rev().tuple_windows::<PointPair>());
        }
    }

    result
}

/// 栅格化多边形
///
/// 该函数实现了多边形的栅格化,包括外环和内环。算法基于GDAL的实现,
/// 使用扫描线算法填充多边形内部,并单独处理边界线。
///
/// # 参数
/// * `first` - 多边形外环线串
/// * `rest` - 多边形内环线串数组
/// * `rasterizer` - 栅格化器实例
///
/// # 类型参数
/// * `Label` - 栅格化时使用的标签类型,需要实现Copy和Add特征
///
/// # 说明
/// 该实现参考了GDAL中的GDALdllImageFilledPolygon算法。
/// 主要步骤:
/// 1. 收集多边形所有顶点
/// 2. 按扫描线逐行处理
/// 3. 计算扫描线与多边形边的交点
/// 4. 对交点配对并填充像素
/// 5. 最后处理多边形边界
pub fn rasterize_polygon<Label>(
    first: &LineString<f64>,
    rest: &[LineString<f64>],
    rasterizer: &mut Rasterizer<Label>,
) where
    Label: Copy + Add<Output = Label>,
{
    // 确保所有线串都是闭合的
    assert!(first.is_closed() && rest.iter().all(|ls| ls.is_closed()));

    // 计算总点数
    let total_points = first.coords_count()
        + rest
            .iter()
            .map(|line_string| line_string.coords_count())
            .sum::<usize>();
    if total_points == 0 {
        return;
    }

    // 用于存储扫描线与多边形边的交点
    let mut xs: Vec<isize> = Vec::with_capacity(total_points);

    // 计算多边形的y坐标范围
    let min_y = y_coordinates(first, rest).min().unwrap().max(0);
    let max_y = y_coordinates(first, rest)
        .max()
        .unwrap()
        .min(rasterizer.height() as isize - 1);
    let min_x = 0;
    let max_x = rasterizer.width() - 1;

    // 获取顺时针排序的点对序列
    let cw_points = into_pointpairs(first, rest);

    // 逐行扫描处理
    for y in min_y..=max_y {
        let dy = 0.5 + (y as f64); // 扫描线的中心高度

        // 处理每个点对(多边形边)
        for (ind1, ind2) in cw_points.iter() {
            let mut dy1 = ind1.y();
            let mut dy2 = ind2.y();

            // 如果边完全在扫描线上方或下方则跳过
            if (dy1 < dy && dy2 < dy) || (dy1 > dy && dy2 > dy) {
                continue;
            }

            // 处理与扫描线的交点
            let (dx1, dx2) = if dy1 < dy2 {
                (ind1.x(), ind2.x())
            } else if dy1 > dy2 {
                std::mem::swap(&mut dy1, &mut dy2);
                (ind2.x(), ind1.x())
            } else {
                // 处理水平边
                if ind1.x() > ind2.x() {
                    let horizontal_x1 = (ind2.x() + 0.5).floor() as isize;
                    let horizontal_x2 = (ind1.x() + 0.5).floor() as isize;
                    if horizontal_x1 > (max_x as isize) || horizontal_x2 <= min_x {
                        continue;
                    }
                    rasterizer.fill_horizontal_line(
                        horizontal_x1 as usize,
                        horizontal_x2 as usize,
                        y as usize,
                    );
                }
                continue;
            };

            // 计算交点x坐标
            if dy < dy2 && dy >= dy1 {
                let intersect = (dy - dy1) * (dx2 - dx1) / (dy2 - dy1) + dx1;
                xs.push((intersect + 0.5).floor() as isize);
            }
        }

        // 对交点排序并填充像素
        xs.sort_unstable();
        for pair in xs[..].chunks_exact(2) {
            let x_start = pair[0].max(min_x);
            let x_end = pair[1].min((max_x + 1) as isize);
            if x_start <= (max_x as isize) && x_end > min_x {
                rasterizer.fill_horizontal_line(x_start as usize, x_end as usize, y as usize);
            }
        }
        xs.clear();
    }

    // 栅格化多边形边界
    once(first)
        .chain(rest)
        .for_each(|ls| ls.rasterize(rasterizer));
}
