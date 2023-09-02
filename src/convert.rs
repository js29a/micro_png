/// convert
///
/// Format conversion utilities.

use std::collections::HashMap;

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

type Cube = Vec<Vec<Vec<u64>>>;

struct Qtz {
    cube: Cube,
    memo: HashMap<u64, Vec<u16>>
}

impl Qtz {
    fn fill_vec(&mut self, mut r: (u16, u16), mut g: (u16, u16), mut b: (u16, u16), c: usize) -> Vec<u16> {
        assert!(c < 3);

        assert!(r.0 <= r.1);
        assert!(g.0 <= g.1);
        assert!(b.0 <= b.1);

        let mut output: Vec<u16> = Vec::new();

        r.0 >>= 8;
        g.0 >>= 8;
        b.0 >>= 8;

        r.1 >>= 8;
        g.1 >>= 8;
        b.1 >>= 8;

        let key: u64 =
            (( c  as u64) << 48) |
            ((r.0 as u64) << 40) |
            ((r.1 as u64) << 32) |
            ((g.0 as u64) << 24) |
            ((g.1 as u64) << 16) |
            ((b.0 as u64) <<  8) |
            ((b.1 as u64) <<  0);

        if let Some(v) = self.memo.get(&key) {
            return v.clone();
        }

        (r.0 ..= r.1).for_each(|rr| {
            (g.0 ..= g.1).for_each(|gg| {
                (b.0 ..= b.1).for_each(|bb| {
                    let cnt = self.cube[rr as usize][gg as usize][bb as usize];
                    (0 .. cnt).for_each(|_| {
                        match c {
                            0 => output.push(rr << 8),
                            1 => output.push(gg << 8),
                            2 => output.push(bb << 8),
                            _ => panic!("bad call of elect_qtz")
                        }
                    });
                });
            });
        });

        self.memo.insert(key, output.clone());

        output
    }
}

// c -> component #, 0 .. 2
fn elect_qtz(r: (u16, u16), g: (u16, u16), b: (u16, u16), c: usize, qtz: &mut Qtz) -> u16 {
    assert!(c < 3);

    assert!(r.0 <= r.1);
    assert!(g.0 <= g.1);
    assert!(b.0 <= b.1);

    //r.0 >>= 8;
    //g.0 >>= 8;
    //b.0 >>= 8;

    //r.1 >>= 8;
    //g.1 >>= 8;
    //b.1 >>= 8;

    let vec = qtz.fill_vec(r, g, b, c);

    //(r.0 ..= r.1).for_each(|rr| {
        //(g.0 ..= g.1).for_each(|gg| {
            //(b.0 ..= b.1).for_each(|bb| {
                //let cnt = qtz.cube[rr as usize][gg as usize][bb as usize];
                //(0 .. cnt).for_each(|_| {
                    //match c {
                        //0 => vec.push(rr << 8),
                        //1 => vec.push(gg << 8),
                        //2 => vec.push(bb << 8),
                        //_ => panic!("bad call of elect_qtz")
                    //}
                //});
            //});
        //});
    //});

    if vec.is_empty() {
        match c {
            0 => ((r.0 as u32 + r.1 as u32) >> 1) as u16,
            1 => ((g.0 as u32 + g.1 as u32) >> 1) as u16,
            2 => ((b.0 as u32 + b.1 as u32) >> 1) as u16,
            _ => panic!("bad call of elect_qtz")
        }
    }
    else {
        vec[vec.len() / 2]
    }
}

fn elect_div(r: (u16, u16), g: (u16, u16), b: (u16, u16)) -> usize {
    let r_dist = r.1 - r.0;
    let g_dist = g.1 - g.0;
    let b_dist = b.1 - b.0;

    if r_dist > g_dist && r_dist > b_dist {
        0
    }
    else if g_dist > b_dist {
        1
    }
    else {
        2
    }
}

fn elect_palette_sub(r: (u16, u16), g: (u16, u16), b: (u16, u16), bits: usize, pal: &mut Vec<RGBA>, qtz: &mut Qtz) {
    if bits == 0 {
        let q_0 = elect_qtz(r, g, b, 0, qtz);
        let q_1 = elect_qtz(r, g, b, 1, qtz);
        let q_2 = elect_qtz(r, g, b, 2, qtz);

        pal.push((
            (q_0 >> 8) as u8,
            (q_1 >> 8) as u8,
            (q_2 >> 8) as u8,
            0xff
        ));
    }
    else {
        let c1 = elect_div(r, g, b);

        let split = elect_qtz(r, g, b, c1, qtz);

        match c1 {
            0 => {
                elect_palette_sub((r.0, split), g, b, bits - 1,  pal, qtz);
                elect_palette_sub((split, r.1), g, b, bits - 1,  pal, qtz);
            },
            1 => {
                elect_palette_sub(r, (g.0, split), b, bits - 1,  pal, qtz);
                elect_palette_sub(r, (split, g.1), b, bits - 1,  pal, qtz);
            },
            2 => {
                elect_palette_sub(r, g, (b.0, split), bits - 1,  pal, qtz);
                elect_palette_sub(r, g, (split, b.1), bits - 1,  pal, qtz);
            },
            _ => panic!("internal elect_palette_sub error")
        }
    }
}

fn elect_palette(orig: &Vec<Vec<RGBA16>>, bits: usize) -> Vec<RGBA> {
    let mut pal: Vec<RGBA> = Vec::new();

    let width = orig[0].len();
    let height = orig.len();

    let mut cube = vec![vec![vec![0_u64; 256]; 256]; 256];

    (0 .. height).for_each(|y| {
        (0 .. width).for_each(|x| {
            let pix = orig[y][x];
            cube[(pix.0 >> 8) as usize][(pix.1 >> 8) as usize][(pix.2 >> 8) as usize] += 1;
        });
    });

    let mut qtz = Qtz {
        cube,
        memo: HashMap::new()
    };

    elect_palette_sub((0, 0xffff), (0, 0xffff), (0, 0xffff), bits, &mut pal, &mut qtz);
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
