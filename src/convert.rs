/// convert
///
/// Format conversion utilities.

use crate::*;

fn to_grayscale(orig: Vec<Vec<RGBA16>>, bits: usize, alpha: bool, gs: Grayscale) -> ImageData {
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

        ImageData::GRAYA(vec![res], gs)
    }
    else {
        let res = orig.iter().map(|line| {
            line.iter().map(|(r, g, b, _a)| {
                let y = ((*r as f32) * 0.299 + (*g as f32) * 0.587 + (*b as f32) * 0.114) / 65535.0;
                ((y * ((1 << bits) as f32)) as u32).clamp(0, (1 << bits) - 1) as u16
            }).collect()
        }).collect();

        ImageData::GRAY(vec![res], gs)
    }
}

// c -> component #, 0 .. 2
fn elect_qtz(_orig: &[Vec<RGBA16>], r: (u16, u16), g: (u16, u16), b: (u16, u16), c: usize,
    cnt: (&[u64], &[u64], &[u64])) -> u16 {
    assert!(c < 3);

    let mut num: u64 = 0;
    let mut den: u64 = 0;

    match c {
        0 =>
            (r.0 ..= r.1).for_each(|p| {
                num += p as u64 * cnt.0[p as usize];
                den += cnt.0[p as usize];
            }),
        1 =>
            (g.0 ..= g.1).for_each(|p| {
                num += p as u64 * cnt.1[p as usize];
                den += cnt.1[p as usize];
            }),
        2 =>
            (b.0 ..= b.1).for_each(|p| {
                num += p as u64 * cnt.2[p as usize];
                den += cnt.2[p as usize];
            }),
        _ => panic!("bad call of elect_qtz")
    };

    //(0 .. height).for_each(|y| {
        //(0 .. width).for_each(|x| {
            //let pix = orig[y][x];

            //match c {
                //0 => {
                    //if pix.0 >= r.0 && pix.0 <= r.1 {
                        //cnt += 1;
                        //avg += pix.0 as u64;
                    //}
                //},
                //1 => {
                    //if pix.1 >= g.0 && pix.1 <= g.1 {
                        //cnt += 1;
                        //avg += pix.1 as u64;
                    //}
                //},
                //2 => {
                    //if pix.2 >= b.0 && pix.2 <= b.1 {
                        //cnt += 1;
                        //avg += pix.2 as u64;
                    //}
                //},
                //_ => panic!("bad call of elect_qtz")
            //}
        //})
    //});

    if den == 0 {
        match c {
            0 => ((r.0 as u32 + r.1 as u32) >> 1) as u16,
            1 => ((g.0 as u32 + g.1 as u32) >> 1) as u16,
            2 => ((b.0 as u32 + b.1 as u32) >> 1) as u16,
            _ => panic!("bad call of elect_qtz")
        }
    }
    else {
        (num / den) as u16
    }

    //let mut vec: Vec<u16> = Vec::new();

    //(0 .. height).for_each(|y| {
        //(0 .. width).for_each(|x| {
            //let pix = orig[y][x];

            //match c {
                //0 =>
                    //if pix.0 >= r.0 && pix.0 <= r.1 {
                        //vec.push(orig[y][x].0)
                    //},
                //1 =>
                    //if pix.1 >= g.0 && pix.1 <= g.1 {
                        //vec.push(orig[y][x].1)
                    //},
                //2 =>
                    //if pix.2 >= b.0 && pix.2 <= b.1 {
                        //vec.push(orig[y][x].2)
                    //},
                //_ => panic!("bad call of elect_qtz")
            //}
        //})
    //});

    //vec.sort();

    //if vec.is_empty() {
        //match c {
            //0 => ((r.0 as u32 + r.1 as u32) >> 1) as u16,
            //1 => ((g.0 as u32 + g.1 as u32) >> 1) as u16,
            //2 => ((b.0 as u32 + b.1 as u32) >> 1) as u16,
            //_ => panic!("bad call of elect_qtz")
        //}
    //}
    //else {
        //vec[vec.len() / 2]
    //}
}

#[allow(clippy::too_many_arguments)]
fn elect_palette_sub(orig: &Vec<Vec<RGBA16>>, r: (u16, u16), g: (u16, u16), b: (u16, u16), bits: usize, c: usize, pal: &mut Vec<RGBA>, cnt: (&[u64], &[u64], &[u64])) {
    assert!(c < 3);

    if bits == 0 {
        //let width = orig[0].len();
        //let height = orig.len();

        //(0 .. height).for_each(|y| {
            //(0 .. width).for_each(|x| {
                //let pix = orig[y][x];

                //if pix.0 >= r.0 && pix.0 <= r.0 &&
                   //pix.1 >= g.0 && pix.1 <= g.1 &&
                   //pix.2 >= b.0 && pix.2 <= b.1 {
                    //r.0 = u16::min(r.0, pix.0);
                    //r.1 = u16::max(r.1, pix.0);
                    //g.0 = u16::min(g.0, pix.1);
                    //g.1 = u16::max(g.1, pix.1);
                    //b.0 = u16::min(b.0, pix.2);
                    //b.1 = u16::max(b.1, pix.2);
                //}
            //});
        //});

        let q_0 = elect_qtz(orig, r, g, b, 0, cnt);
        let q_1 = elect_qtz(orig, r, g, b, 1, cnt);
        let q_2 = elect_qtz(orig, r, g, b, 2, cnt);

        //let q_0 = (r.0 as u32 + r.1 as u32) >> 1;
        //let q_1 = (g.0 as u32 + g.1 as u32) >> 1;
        //let q_2 = (b.0 as u32 + b.1 as u32) >> 1;

        pal.push((
            (q_0 >> 8) as u8,
            (q_1 >> 8) as u8,
            (q_2 >> 8) as u8,
            0xff
        ));
    }
    else {
        let split = elect_qtz(orig, r, g, b, c, cnt);

        match c {
            0 => {
                elect_palette_sub(orig, (r.0, split), g, b, bits - 1, (c + 1) % 3, pal, cnt);
                elect_palette_sub(orig, (split, r.1), g, b, bits - 1, (c + 1) % 3, pal, cnt);
            },
            1 => {
                elect_palette_sub(orig, r, (g.0, split), b, bits - 1, (c + 1) % 3, pal, cnt);
                elect_palette_sub(orig, r, (split, g.1), b, bits - 1, (c + 1) % 3, pal, cnt);
            },
            2 => {
                elect_palette_sub(orig, r, g, (b.0, split), bits - 1, (c + 1) % 3, pal, cnt);
                elect_palette_sub(orig, r, g, (split, b.1), bits - 1, (c + 1) % 3, pal, cnt);
            },
            _ => panic!("internal elect_palette_sub error")
        }
    }
}

fn elect_palette(orig: &Vec<Vec<RGBA16>>, bits: usize) -> Vec<RGBA> {
    let mut pal: Vec<RGBA> = Vec::new();

    let width = orig[0].len();
    let height = orig.len();

    let mut cnt_a = vec![0_u64; 0xffff + 1];
    let mut cnt_g = vec![0_u64; 0xffff + 1];
    let mut cnt_b = vec![0_u64; 0xffff + 1];

    (0 .. height).for_each(|y| {
        (0 .. width).for_each(|x| {
            let pix = orig[y][x];
            cnt_a[pix.0 as usize] += 1;
            cnt_g[pix.1 as usize] += 1;
            cnt_b[pix.2 as usize] += 1;
        });
    });

    elect_palette_sub(orig, (0, 0xffff), (0, 0xffff), (0, 0xffff), bits, 0, &mut pal,
        (&cnt_a[..], &cnt_g[..], &cnt_b[..]));
    assert_eq!(pal.len(), 1 << bits);
    pal
}

// TODO strip alpha from palette for NDX no A modes
fn to_indexed(orig: Vec<Vec<RGBA16>>, bits: usize, _alpha: bool, pt: Palette) -> ImageData {
    let pal = elect_palette(&orig, bits);
    assert_eq!(pal.len(), 1 << bits);

    let width = orig[0].len();
    let height = orig.len();

    // TODO alpha level
    let mut closest: Vec<Vec<Vec<Option<u8>>>> = vec![vec![vec![None; 256]; 256]; 256];

    let mut res = vec![vec![0_u8; width]; height];

    (0 .. height).for_each(|y| {
        (0 .. width).for_each(|x| {
            let r = (orig[y][x].0 >> 8) as i32;
            let g = (orig[y][x].1 >> 8) as i32;
            let b = (orig[y][x].2 >> 8) as i32;
            let _a = (orig[y][x].3 >> 8) as i32;

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

    ImageData::NDXA(vec![res], pal, pt)
}

/// Convert RGBA HDR to any format.
pub fn convert_hdr(dest: ColorType, orig: Vec<Vec<RGBA16>>) -> Result<ImageData, String> {
    // TODO verify if dims ok (?)

    if orig.is_empty() {
        return Err("empty image for convert_hdr".to_string())
    }

    let w = orig[0].len();

    if w == 0 {
        return Err("empty image for convert_hdr".to_string())
    }

    let mut err: Option<String> = None;

    orig.iter().for_each(|row| {
        if row.len() != w {
            err = Some("inconsistent rows - convert_hdr".to_string())
        }
    });

    if let Some(e) = err {
        return Err(e);
    }

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
        ColorType::GRAY(Grayscale::G1) => Ok(to_grayscale(orig, 1, false, Grayscale::G1)),
        ColorType::GRAY(Grayscale::G2) => Ok(to_grayscale(orig, 2, false, Grayscale::G2)),
        ColorType::GRAY(Grayscale::G4) => Ok(to_grayscale(orig, 4, false, Grayscale::G4)),
        ColorType::GRAY(Grayscale::G8) => Ok(to_grayscale(orig, 8, false, Grayscale::G8)),
        ColorType::GRAY(Grayscale::G16) => Ok(to_grayscale(orig, 16, false, Grayscale::G16)),
        ColorType::GRAYA(Grayscale::G8) => Ok(to_grayscale(orig, 8, true, Grayscale::G8)),
        ColorType::GRAYA(Grayscale::G16) => Ok(to_grayscale(orig, 16, true, Grayscale::G16)),
        ColorType::NDXA(Palette::P1) => Ok(to_indexed(orig, 1, true, Palette::P1)),
        ColorType::NDXA(Palette::P2) => Ok(to_indexed(orig, 2, true, Palette::P2)),
        ColorType::NDXA(Palette::P4) => Ok(to_indexed(orig, 4, true, Palette::P4)),
        ColorType::NDXA(Palette::P8) => Ok(to_indexed(orig, 8, true, Palette::P8)),
        ColorType::NDX(Palette::P1) => Ok(to_indexed(orig, 1, false, Palette::P1)),
        ColorType::NDX(Palette::P2) => Ok(to_indexed(orig, 2, false, Palette::P2)),
        ColorType::NDX(Palette::P4) => Ok(to_indexed(orig, 4, false, Palette::P4)),
        ColorType::NDX(Palette::P8) => Ok(to_indexed(orig, 8, false, Palette::P8)),
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
