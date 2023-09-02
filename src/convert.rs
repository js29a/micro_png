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
                    ((y * ((1 << bits) as f32)) as u32).clamp(0, (1 << bits) - 1) as u16,
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
                ((y * ((1 << bits) as f32)) as u32).clamp(0, (1 << bits) - 1) as u16
            }).collect()
        }).collect();

        Ok(ImageData::GRAY(vec![res], gs))
    }
}

fn elect_palette(orig: &Vec<Vec<RGBA16>>, bits: usize) -> Result<Vec<RGBA>, String> {
    let mut pal: Vec<RGBA> = vec![(0, 0, 0, 0); 1 << bits];

// 3 3 2 alpha 0xff
    (0 .. 1 << bits).for_each(|n| {
        let r = (n & 7) << 5;
        let g = ((n >> 3) & 7) << 5;
        let b = ((n >> 6) & 3) << 6;
        pal[n] = (
            r as u8,
            g as u8,
            b as u8,
            0xff_u8
        );
    });

    Ok(pal)
}

// TODO strip alpha from palette for NDX no A modes
fn to_indexed(orig: Vec<Vec<RGBA16>>, bits: usize, alpha: bool, pt: Palette) -> Result<ImageData, String> {
    let pal = elect_palette(&orig, bits)?;
    assert_eq!(pal.len(), 1 << bits);

    let width = orig[0].len();
    let height = orig.len();

    // TODO alpha level
    //let mut closest: [[[Option<u8>; 256]; 256]; 256] = [[[None; 256]; 256]; 256];
    let mut closest: Vec<Vec<Vec<Option<u8>>>> = vec![vec![vec![None; 256]; 256]; 256];

    let mut res = vec![vec![0_u8; width]; height];

    (0 .. height).for_each(|y| {
        (0 .. width).for_each(|x| {
            let r = (orig[y][x].0 >> 8) as i32;
            let g = (orig[y][x].1 >> 8) as i32;
            let b = (orig[y][x].1 >> 8) as i32;
            let _a = (orig[y][x].1 >> 8) as i32;

            let ndx = match closest[r as usize][g as usize][b as usize] {
                Some(k) => k,
                _ => {
                    let mut best = 0_u8;
                    let mut best_dist =
                        (r - pal[0].0 as i32) * (r - pal[0].0 as i32) +
                        (g - pal[0].1 as i32) * (g - pal[0].1 as i32) +
                        (b - pal[0].2 as i32) * (b - pal[0].2 as i32);

                    (1 .. pal.len()).for_each(|n| {
                        let cur_dist =
                            (r - pal[n].0 as i32) * (r - pal[n].0 as i32) +
                            (g - pal[n].1 as i32) * (g - pal[n].1 as i32) +
                            (b - pal[n].2 as i32) * (b - pal[n].2 as i32);

                        if cur_dist < best_dist {
                            best = n as u8;
                            best_dist = cur_dist;
                        }
                    });

                    closest[r as usize][g as usize][b as usize] = Some(best);

                    best
                },
            };

            res[y][x] = ndx;
        });
    });

    Ok(ImageData::NDXA(vec![res], pal, pt))
}

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
        ColorType::NDXA(Palette::P1) => to_indexed(orig, 1, true, Palette::P1),
        ColorType::NDXA(Palette::P2) => to_indexed(orig, 2, true, Palette::P2),
        ColorType::NDXA(Palette::P4) => to_indexed(orig, 4, true, Palette::P4),
        ColorType::NDXA(Palette::P8) => to_indexed(orig, 8, true, Palette::P8),
        ColorType::NDX(Palette::P1) => to_indexed(orig, 1, false, Palette::P1),
        ColorType::NDX(Palette::P2) => to_indexed(orig, 2, false, Palette::P2),
        ColorType::NDX(Palette::P4) => to_indexed(orig, 4, false, Palette::P4),
        ColorType::NDX(Palette::P8) => to_indexed(orig, 8, false, Palette::P8),
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
            //ColorType::RGBA16, // remove alpha
            //ColorType::RGB16, // remove alpha
            //ColorType::RGBA,
            //ColorType::RGB,
            //ColorType::GRAYA(Grayscale::G16),
            //ColorType::GRAYA(Grayscale::G8),
            //ColorType::GRAY(Grayscale::G16),
            //ColorType::GRAY(Grayscale::G8),
            //ColorType::GRAY(Grayscale::G4),
            //ColorType::GRAY(Grayscale::G2),
            //ColorType::GRAY(Grayscale::G1),
            ColorType::NDX(Palette::P1),
            ColorType::NDX(Palette::P2),
            ColorType::NDX(Palette::P4),
            ColorType::NDX(Palette::P8),
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
