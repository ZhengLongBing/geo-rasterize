use std::fmt::Debug;

use anyhow::Result;
use gdal::DriverManager;
use geo::algorithm::{
    coords_iter::CoordsIter,
    map_coords::{MapCoords, MapCoordsInPlace},
};
use geo::Coord;
use ndarray::Array2;
use num_traits::{Num, NumCast};

use super::{MergeAlgorithm, Rasterize, Rasterizer};

fn to_float<T>(coords: Coord<T>) -> Coord<f64>
where
    T: Into<f64> + Copy + NumCast + Debug + PartialOrd + Num, // T必须可复制且可转换为f64
{
    Coord {
        x: coords.x.into(),
        y: coords.y.into(),
    }
}

/// Use `gdal`'s rasterizer to rasterize some shape into a
/// (widith, height) window of u8.
pub fn gdal_rasterize<Coord, InputShape, ShapeAsF64>(
    width: usize,
    height: usize,
    shapes: &[InputShape],
    algorithm: MergeAlgorithm,
) -> Result<Array2<u8>>
where
    InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
    ShapeAsF64: Rasterize<u8>
        + for<'a> CoordsIter<Scalar = f64>
        + Into<geo::Geometry<f64>>
        + MapCoordsInPlace<f64>,
    Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
{
    use gdal::{
        raster::{rasterize, RasterizeOptions},
        vector::ToGdal,
    };

    let driver = DriverManager::get_driver_by_name("MEM")?;
    let mut ds = driver.create_with_band_type::<u8, &str>("some_filename", width, height, 1)?;
    let options = RasterizeOptions {
        all_touched: true,
        merge_algorithm: match algorithm {
            MergeAlgorithm::Replace => gdal::raster::MergeAlgorithm::Replace,
            MergeAlgorithm::Add => gdal::raster::MergeAlgorithm::Add,
        },
        ..Default::default()
    };

    let mut gdal_shapes = Vec::new();
    for shape in shapes {
        let float = shape.map_coords(to_float);
        let all_finite = float
            .coords_iter()
            .all(|coordinate| coordinate.x.is_finite() && coordinate.y.is_finite());

        assert!(all_finite);
        let geom: geo::Geometry<f64> = float.into();
        let gdal_geom = geom.to_gdal()?;
        gdal_shapes.push(gdal_geom);
    }
    let burn_values = vec![1.0; gdal_shapes.len()];
    rasterize(&mut ds, &[1], &gdal_shapes, &burn_values, Some(options))?;
    let data = ds
        .rasterband(1)?
        .read_as((0, 0), (width, height), (width, height), None)?;

    Ok(Array2::from_shape_vec(
        (height, width),
        data.data().to_vec(),
    )?)
}

pub fn compare<Coord, InputShape, ShapeAsF64>(
    width: usize,
    height: usize,
    shapes: &[InputShape],
    algorithm: MergeAlgorithm,
) -> Result<(Array2<u8>, Array2<u8>)>
where
    InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
    ShapeAsF64: Rasterize<u8>
        + for<'a> CoordsIter<Scalar = f64>
        + Into<geo::Geometry<f64>>
        + MapCoordsInPlace<f64>,
    Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
{
    let mut r = Rasterizer::new(width, height, None, algorithm, 0u8);
    for shape in shapes.iter() {
        r.rasterize(shape, 1u8)?;
    }
    let actual = r.finish();

    let expected = gdal_rasterize(width, height, shapes, algorithm)?;
    if actual != expected {
        println!("{}\n\n{}", actual, expected);
    }
    Ok((actual, expected))
}
