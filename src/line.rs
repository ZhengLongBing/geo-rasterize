use std::ops::Add;

use geo::Line;

use crate::Rasterizer;

/// 将线段栅格化为像素
///
/// # 类型参数
/// * `Label` - 栅格化后的标签类型，必须实现Copy和Add特征
///
/// # 参数
/// * `line` - 要栅格化的线段，坐标类型为f64
/// * `rasterizer` - 用于执行栅格化的栅格化器实例
pub fn rasterize_line<Label>(line: &Line<f64>, rasterizer: &mut Rasterizer<Label>)
where
    Label: Copy + Add<Output = Label>,
{
    // 获取栅格化器的尺寸
    let width = rasterizer.width() as f64;
    let height = rasterizer.height() as f64;

    // 检查线段是否完全在栅格范围外
    if (line.start.y < 0. && line.end.y < 0.)
        || (line.start.y > height && line.end.y > height)
        || (line.start.x < 0. && line.end.x < 0.)
        || (line.start.x > width && line.end.x > width)
    {
        return;
    }

    // 确保线段从左向右处理，如果起点在终点右侧则交换两点
    let line = if line.start.x > line.end.x {
        Line::new(line.end, line.start)
    } else {
        *line
    };

    // 定义判断垂直和水平线段的阈值
    const THRESHOLD: f64 = 0.01;
    // 判断是否为垂直线段
    let is_vertical = (line.start.x.floor() == line.end.x.floor()) || line.dx().abs() < THRESHOLD;
    // 判断是否为水平线段
    let is_horizontal = (line.start.y.floor() == line.end.y.floor()) || line.dy().abs() < THRESHOLD;

    if is_vertical {
        // 处理垂直线段
        // 确保y_start小于y_end
        let (y_start, y_end) = if line.start.y > line.end.y {
            (line.end.y, line.start.y)
        } else {
            (line.start.y, line.end.y)
        };
        let x = line.end.x;

        // 检查x坐标是否在有效范围内
        let ix = x.floor() as isize;
        if ix < 0 || ix >= (rasterizer.width() as isize) {
            return;
        }

        // 将y坐标限制在有效范围内
        let y_start = (y_start.floor() as usize).clamp(0, rasterizer.height() - 1);
        let y_end = (y_end.floor() as usize).clamp(0, rasterizer.height() - 1);
        rasterizer.fill_vertical_line_no_repeat(ix as usize, y_start, y_end);
    } else if is_horizontal {
        // 处理水平线段
        // 确保x_start小于x_end
        let (x_start, x_end) = if line.start.x > line.end.x {
            (line.end.x, line.start.x)
        } else {
            (line.start.x, line.end.x)
        };
        let y = line.start.y;

        // 检查y坐标是否在有效范围内
        let iy = y.floor() as isize;
        if iy < 0 || iy >= (rasterizer.height() as isize) {
            return;
        }

        // 将x坐标限制在有效范围内
        let x_start = (x_start.floor() as usize).clamp(0, rasterizer.width() - 1);
        let x_end = (x_end.floor() as usize).clamp(0, rasterizer.width() - 1);
        rasterizer.fill_horizontal_line_no_repeat(x_start, x_end, iy as usize);
    } else {
        // 处理一般斜线
        let slope = line.slope(); // 计算斜率
        let (mut x_start, mut y_start) = line.start.x_y();
        let (mut x_end, mut y_end) = line.end.x_y();

        // 在x方向裁剪线段
        if x_end > width {
            y_end -= (x_end - width) * slope;
            x_end = width;
        }
        if x_start < 0. {
            y_start += (0. - x_start) * slope;
            x_start = 0.;
        }

        // 在y方向裁剪线段
        if y_end > y_start {
            // 线段向上倾斜
            if y_start < 0. {
                let x_diff = -y_start / slope;
                x_start += x_diff;
                y_start = 0.;
            }
            if y_end >= height {
                x_end += (y_end - height) / slope;
            }
        } else {
            // 线段向下倾斜
            if y_start >= height {
                let x_diff = (height - y_start) / slope;
                x_start += x_diff;
                y_start = height;
            }
            if y_end < 0. {
                x_end -= y_end / slope;
            }
        }

        // 逐像素处理线段
        while (x_start >= 0.) && (x_start < x_end) {
            let ix = x_start.floor() as isize;
            let iy = y_start.floor() as isize;

            // 如果当前像素在有效范围内，进行填充
            if iy >= 0 && ((iy as usize) < rasterizer.height()) {
                rasterizer.fill_horizontal_line_no_repeat(ix as usize, ix as usize, iy as usize);
            }

            // 计算下一个像素的步进量
            let mut x_step = (x_start + 1.).floor() - x_start;
            let mut y_step = x_step * slope;

            // 根据斜率调整步进量，确保正确处理扫描线
            if ((y_start + y_step).floor() as isize) == iy {
                // 保持在当前扫描线上
            } else if slope < 0. {
                // 负斜率情况
                const STEP_THRESHOLD: f64 = -0.000000001;
                y_step = ((iy as f64) - y_start).min(STEP_THRESHOLD);
                x_step = y_step / slope;
            } else {
                // 正斜率情况
                const STEP_THRESHOLD: f64 = 0.000000001;
                y_step = (((iy + 1) as f64) - y_start).max(STEP_THRESHOLD);
                x_step = y_step / slope;
            }

            // 更新坐标位置
            x_start += x_step;
            y_start += y_step;
        }
    }
}
