use geo::{
    map_coords::MapCoords, Coord, Geometry, Line, LineString, MultiLineString, MultiPoint,
    MultiPolygon, Point, Polygon, Rect, Triangle,
};
use pretty_assertions::assert_eq;
use proptest::prelude::*;

use crate::tests::utils::compare;
use crate::MergeAlgorithm;

// 生成随机点
#[rustfmt::skip]
prop_compose! {
    fn arb_point()(x in -2.0..19., y in -2.0..21.0) -> Point<f64> {
        Point::new(x, y)
    }
}

// 生成随机线段
#[rustfmt::skip]
prop_compose! {
    fn arb_line()(start in arb_point(), end in arb_point()) -> Line<f64> {
        Line::new(start, end)
    }
}

// 生成随机折线,包含2-20个点
#[rustfmt::skip]
prop_compose! {
    fn arb_linestring()(points in prop::collection::vec(arb_point(), 2..20)) -> LineString<f64> {
        points.into()
    }
}

// 生成随机矩形,通过中心点和宽高定义
#[rustfmt::skip]
prop_compose! {
    fn arb_rect()(center in arb_point(),
                  width in 0.0..20., height in 0.0..20.) -> Rect<f64> {
        let min: Coord<f64> = (
            center.x() - width / 2.,
            center.y() - height / 2.).into();
        let max: Coord<f64> = (
            center.x() + width / 2.,
            center.y() + height/2.).into();
        Rect::new(min, max)
    }
}

// 生成随机环形线,返回中心点和环形线段
#[rustfmt::skip]
prop_compose! {
    fn arb_linearring()(center in arb_point(),
                       exterior_points in 3..17,
                       radius in 0.000001..15.0) -> (Point<f64>, LineString<f64>) {
        let angles = (0..exterior_points)
            .map(|idx| 2.0 * std::f64::consts::PI * (idx as f64) / (exterior_points as f64));
        let points: Vec<Coord<f64>> = angles
            .map(|angle_rad| angle_rad.sin_cos())
            .map(|(sin, cos)| Coord {
                x: center.x() + radius * cos,
                y: center.y() + radius * sin,
            })
            .collect();
        (center, LineString(points))
    }
}

/// 缩放环形线
///
/// # 参数
/// * `center` - 环形线的中心点
/// * `ring` - 要缩放的环形线
/// * `scale_factor` - 缩放因子
fn shrink_ring(center: &Point<f64>, ring: &LineString<f64>, scale_factor: f64) -> LineString<f64> {
    let cx = center.x();
    let cy = center.y();
    use euclid::{Point2D, Transform2D, UnknownUnit, Vector2D};
    let transform: Transform2D<f64, UnknownUnit, UnknownUnit> = Transform2D::identity()
        .then_translate(Vector2D::new(-cx, -cy))
        .then_scale(scale_factor, scale_factor)
        .then_translate(Vector2D::new(cx, cy));
    ring.map_coords(|Coord { x, y }| {
        transform
            .transform_point(Point2D::new(x, y))
            .to_tuple()
            .into()
    })
}

// 生成随机多边形,可选是否包含内部空洞
#[rustfmt::skip]
prop_compose! {
    fn arb_poly()(center_exterior in arb_linearring(),
                  include_hole in proptest::bool::ANY,
                  hole_scale_factor in 0.01..0.9) -> Polygon<f64> {
        let (center, exterior) = center_exterior;
        let holes = if include_hole {
            vec![shrink_ring(&center, &exterior, hole_scale_factor)]
        } else {
            vec![]
        };
        Polygon::new(exterior, holes)
    }
}

// 生成随机三角形
#[rustfmt::skip]
prop_compose! {
    fn arb_triangle()(a in arb_point(),
                      b in arb_point(),
                      c in arb_point()) -> Triangle<f64> {
        Triangle(a.0, b.0, c.0)
    }
}

// 生成随机点集合(0-5个点)
#[rustfmt::skip]
prop_compose! {
    fn arb_multipoint()(points in proptest::collection::vec(arb_point(), 0..5)) -> MultiPoint<f64>{
        MultiPoint(points)
    }
}

// 生成随机线段集合(0-5条线)
#[rustfmt::skip]
prop_compose! {
    fn arb_multiline()(lines in proptest::collection::vec(arb_linestring(), 0..5)) -> MultiLineString<f64>{
        MultiLineString(lines)
    }
}

// 生成随机多边形集合(0-5个多边形)
#[rustfmt::skip]
prop_compose! {
    fn arb_multipoly()(polys in proptest::collection::vec(arb_poly(), 0..5)) -> MultiPolygon<f64>{
        MultiPolygon(polys)
    }
}

/// 生成随机几何图形
fn arb_geo() -> impl Strategy<Value = Geometry<f64>> {
    prop_oneof![
        arb_point().prop_map(Geometry::Point),
        arb_line().prop_map(Geometry::Line),
        arb_poly().prop_map(Geometry::Polygon),
        arb_linestring().prop_map(Geometry::LineString),
        arb_rect().prop_map(Geometry::Rect),
        arb_triangle().prop_map(Geometry::Triangle),
        arb_multipoint().prop_map(Geometry::MultiPoint),
        arb_multiline().prop_map(Geometry::MultiLineString),
        arb_multipoly().prop_map(Geometry::MultiPolygon)
    ]
}

// 测试栅格化结果是否与GDAL一致
#[rustfmt::skip]
proptest! {
    #[test]
    fn match_gdal(shapes in proptest::collection::vec(arb_geo(), 1..5)) {
        let (actual, expected) = compare(17, 19, &shapes, MergeAlgorithm::Replace).unwrap();
        assert_eq!(actual, expected);
    }
}
