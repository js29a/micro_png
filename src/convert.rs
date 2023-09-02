/// convert
///
/// Format conversion utilities.

use crate::*;

fn to_grayscale(orig: Vec<Vec<RGBA16>>, bits: usize, alpha: bool, gs: Grayscale) -> Result<ImageData, String> {
    //  0.299 0.587 0.114

    if alpha {
        let res = orig.iter().map(|line| {
            line.iter().map(|(r, g, b, a)| {
                let y = ((*r as f32) * 0.299 + (*g as f32) * 0.587 + (*b as f32) * 0.114) / 65535.0;
                (
                    ((y * ((1 << bits) as f32)) as u32).clamp(0, 1 << bits) as u16,
                    *a / ( if bits == 8  { 256 } else { 1 })
                )
            }).collect()
        }).collect();

        Ok(ImageData::GRAYA(vec![res], gs))
    }
    else {
        let res = orig.iter().map(|line| {
            line.iter().map(|(r, g, b, _a)| {
                let y = ((*r as f32) * 0.299 + (*g as f32) * 0.587 + (*b as f32) * 0.114) / 65535.0;
                ((y * ((1 << bits) as f32)) as u32).clamp(0, 1 << bits) as u16
            }).collect()
        }).collect();

        Ok(ImageData::GRAY(vec![res], gs))
    }
}

//fn to_palette(orig: Vec<Vec<RGBA16>>, bits: usize) -> Result<ImageData, String> {
    //Err("to be done".to_string())
//}

/// Convert RGBA HDR to any format.
pub fn convert_hdr(dest: ColorType, orig: Vec<Vec<RGBA16>>) -> Result<ImageData, String> {
    // TODO verify if dims ok (?)

    match dest {
        ColorType::RGBA16 => 
            Ok(
                ImageData::RGBA16(vec![orig])
            ),
        ColorType::RGB16 => {
            Ok(
                ImageData::RGB16(
                    vec![
                        orig.iter().map(|line| {
                            line.iter().map(|(r, g, b, _a)| {
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
                            line.iter().map(|(r, g, b, _a)| {
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
        ColorType::GRAY(Grayscale::G1) => to_grayscale(orig, 1, false, Grayscale::G1),
        ColorType::GRAY(Grayscale::G2) => to_grayscale(orig, 2, false, Grayscale::G2),
        ColorType::GRAY(Grayscale::G4) => to_grayscale(orig, 4, false, Grayscale::G4),
        ColorType::GRAY(Grayscale::G8) => to_grayscale(orig, 8, false, Grayscale::G8),
        ColorType::GRAY(Grayscale::G16) => to_grayscale(orig, 16, false, Grayscale::G16),
        ColorType::GRAYA(Grayscale::G8) => to_grayscale(orig, 8, true, Grayscale::G8),
        ColorType::GRAYA(Grayscale::G16) => to_grayscale(orig, 16, true, Grayscale::G16),
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
            ColorType::RGBA16, // remove alpha
            ColorType::RGB16, // remove alpha
            ColorType::RGBA,
            ColorType::RGB,
            ColorType::GRAYA(Grayscale::G16),
            ColorType::GRAYA(Grayscale::G8),
            ColorType::GRAY(Grayscale::G16),
            ColorType::GRAY(Grayscale::G8),
            ColorType::GRAY(Grayscale::G4),
            ColorType::GRAY(Grayscale::G2),
            ColorType::GRAY(Grayscale::G1),
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
