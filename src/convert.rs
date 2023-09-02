///! Format conversion utilities.

use crate::*;

fn to_grayscale(orig: Vec<Vec<RGBA16>>, bits: usize, alpha: bool) -> Result<ImageData, String> {
    Err("to be done".to_string())
}

//fn to_palette(orig: Vec<Vec<RGBA16>>, bits: usize) -> Result<ImageData, String> {
    //Err("to be done".to_string())
//}

pub fn convert_hdr(dest: ColorType, orig: Vec<Vec<RGBA16>>) -> Result<ImageData, String> {
    // TODO verify if dims ok (?)

    let width = orig.len();
    let height = orig[0].len();

    match dest {
        ColorType::RGB16 => {
            Ok(
                ImageData::RGB16(
                    vec![
                        orig.iter().map(|line| {
                            line.iter().map(|(r, g, b, a)| {
                                (*r, *g, *b)
                            }).collect()
                        }).collect()
                    ]
                )
            )
        },
        ColorType::RGBA => {
            Ok(
                ImageData::RGBA(
                    vec![
                        orig.iter().map(|line| {
                            line.iter().map(|(r, g, b, a)| {
                                (
                                    (*r >> 8) as u8,
                                    (*g >> 8) as u8,
                                    (*b >> 8) as u8,
                                    (*a >> 8) as u8
                                )
                            }).collect()
                        }).collect()
                    ]
                )
            )
        },
        ColorType::RGB => {
            Ok(
                ImageData::RGB(
                    vec![
                        orig.iter().map(|line| {
                            line.iter().map(|(r, g, b, a)| {
                                (
                                    (*r >> 8) as u8,
                                    (*g >> 8) as u8,
                                    (*b >> 8) as u8
                                )
                            }).collect()
                        }).collect()
                    ]
                )
            )
        },
        ColorType::GRAY(Grayscale::G1) => to_grayscale(orig, 1, false),
        ColorType::GRAY(Grayscale::G2) => to_grayscale(orig, 2, false),
        ColorType::GRAY(Grayscale::G4) => to_grayscale(orig, 4, false),
        ColorType::GRAY(Grayscale::G8) => to_grayscale(orig, 8, false),
        ColorType::GRAY(Grayscale::G16) => to_grayscale(orig, 8, false),
        ColorType::GRAYA(Grayscale::G8) => to_grayscale(orig, 4, true),
        ColorType::GRAYA(Grayscale::G16) => to_grayscale(orig, 8, true),
        _ => Err("to be done".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIG: &str = "fixtures/srce.png";

    #[test]
    pub fn test_convert() {
        let targets = vec![
            ColorType::RGB16, // remove alpha
            ColorType::RGBA,
            ColorType::RGB,
        ];

        let orig = read_png(ORIG).unwrap();

        targets.iter().for_each(|target| {
            let data = convert_hdr(*target, orig.data()).unwrap();

            build_apng(
                APNGBuilder::new(format!("tmp/test-conv-{target:?}.png").as_str(), data)
                    .set_filter(Filter::Paeth)
            ).unwrap();
        });
    }
}
