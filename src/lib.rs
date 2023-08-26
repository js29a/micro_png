use std::fs::File;
use std::io::{Write, Read};
use std::iter::zip;
use std::fs;

use std::collections::HashMap;

use async_std::task::spawn;
use futures::future::join_all;
//use futures_executor;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use flate2::read::ZlibDecoder;
use crc32fast::Hasher;
use quicklz::{compress, CompressionLevel};

pub type RGB = (u8, u8, u8);
pub type RGBA = (u8, u8, u8, u8);

pub type RGB16 = (u16, u16, u16);
pub type RGBA16 = (u16, u16, u16, u16);

pub type NDX = u8;

const ADAM_7: [usize; 64] = [
    1, 6, 4, 6, 2, 6, 4, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    5, 6, 5, 6, 5, 6, 5, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    3, 6, 4, 6, 3, 6, 4, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    5, 6, 5, 6, 5, 6, 5, 6,
    7, 7, 7, 7, 7, 7, 7, 7
];

const ADAM_7_SZ: usize = 8;

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum PkEst {
    None,
    Sub,
    Up,
    Avg,
    Paeth
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum ColorType {
    RGB16,
    RGBA16,
    RGB,
    RGBA,
    NDX,
    NDXA
}

#[derive(Eq, Hash, PartialEq, Debug, Clone)]
pub enum ImageData {
    RGBA16(Vec<Vec<Vec<RGBA16>>>),
    RGB16(Vec<Vec<Vec<RGB16>>>),
    RGBA(Vec<Vec<Vec<RGBA>>>),
    RGB(Vec<Vec<Vec<RGB>>>),
    NDX(Vec<Vec<Vec<NDX>>>, Vec<RGB>),
    NDXA(Vec<Vec<Vec<NDX>>>, Vec<RGBA>)
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Image {
    color_type: ColorType,
    width: usize,
    height: usize,
    data: Vec<Vec<RGBA16>>,
    meta: HashMap<String, String>,
    raw: ImageData
}

impl Image {
    pub fn color_type(&self) -> ColorType {
        self.color_type
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> Vec<Vec<RGBA16>> {
        self.data.clone()
    }

    pub fn raw(&self) -> &ImageData {
        &self.raw
    }

    pub fn meta(&self) -> &HashMap<String, String> {
        &self.meta
    }
}

pub type APNGProgress = fn (cur: usize, total: usize, descr: &String);

pub type Pred = fn (line: &[RGBA16], above: &[RGBA16], color_type: ColorType) -> Vec<u8>;

fn sub(a: u8, b: u8) -> u8 {
    (a as i16 - b as i16) as u8
}

fn add(a: u8, b: u8) -> u8 {
    (a as i16 + b as i16) as u8
}

fn png_chunk(data: &[u8]) -> Vec<u8> {
    let mut res: Vec<u8> = vec![];
    res.extend((data.len() as u32 - 4_u32).to_be_bytes());
    res.extend(data);

    let mut crc = Hasher::new();
    crc.update(data);
    res.extend((crc.finalize()).to_be_bytes());

    res
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i32 + b as i32 - c as i32;
    let pa = (p - a as i32).abs();
    let pb = (p - b as i32).abs();
    let pc = (p - c as i32).abs();

    if pa <= pb && pa <= pc {
        a
    }
    else if pb <= pc {
        b
    }
    else {
        c
    }
}

fn png_none(row: &[RGBA16], _above: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();

    res.push(0_u8);

    res.extend(row.iter().map(|pix| {
        match color_type {
            ColorType::RGBA16 => vec![
                (pix.0 >> 8) as u8,
                (pix.0 & 0xff) as u8,
                (pix.1 >> 8) as u8,
                (pix.1 & 0xff) as u8,
                (pix.2 >> 8) as u8,
                (pix.2 & 0xff) as u8,
                (pix.3 >> 8) as u8,
                (pix.3 & 0xff) as u8,
            ],
            ColorType::RGB16 =>  vec![
                (pix.0 >> 8) as u8,
                (pix.0 & 0xff) as u8,
                (pix.1 >> 8) as u8,
                (pix.1 & 0xff) as u8,
                (pix.2 >> 8) as u8,
                (pix.2 & 0xff) as u8
            ],
            ColorType::RGBA => vec![
                (pix.0 >> 8) as u8,
                (pix.1 >> 8) as u8,
                (pix.2 >> 8) as u8,
                (pix.3 >> 8) as u8
            ],
            ColorType::RGB =>  vec![
                (pix.0 >> 8) as u8,
                (pix.1 >> 8) as u8,
                (pix.2 >> 8) as u8
            ],
            ColorType::NDXA | ColorType::NDX => vec![pix.0 as u8],
        }
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_sub(row: &[RGBA16], _above: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();
    let mut prev: RGBA16 = (0, 0, 0, 0);

    res.push(1_u8);

    res.extend(row.iter().map(|pix| {
        let r = match color_type {
            ColorType::RGBA16 => {
                vec![
                    sub((pix.0 >> 8) as u8, (prev.0 >> 8) as u8),
                    sub((pix.0 & 0xff) as u8, (prev.0 & 0xff) as u8),
                    sub((pix.1 >> 8) as u8, (prev.1 >> 8) as u8),
                    sub((pix.1 & 0xff) as u8, (prev.1 & 0xff) as u8),
                    sub((pix.2 >> 8) as u8, (prev.2 >> 8) as u8),
                    sub((pix.2 & 0xff) as u8, (prev.2 & 0xff) as u8),
                    sub((pix.3 >> 8) as u8, (prev.3 >> 8) as u8),
                    sub((pix.3 & 0xff) as u8, (prev.3 & 0xff) as u8),
                ]
            },
            ColorType::RGB16 =>  vec![
                    sub((pix.0 >> 8) as u8, (prev.0 >> 8) as u8),
                    sub((pix.0 & 0xff) as u8, (prev.0 & 0xff) as u8),
                    sub((pix.1 >> 8) as u8, (prev.1 >> 8) as u8),
                    sub((pix.1 & 0xff) as u8, (prev.1 & 0xff) as u8),
                    sub((pix.2 >> 8) as u8, (prev.2 >> 8) as u8),
                    sub((pix.2 & 0xff) as u8, (prev.2 & 0xff) as u8),
            ],
            ColorType::RGBA => vec![
                sub((pix.0 >> 8) as u8, (prev.0 >> 8) as u8),
                sub((pix.1 >> 8) as u8, (prev.1 >> 8) as u8),
                sub((pix.2 >> 8) as u8, (prev.2 >> 8) as u8),
                sub((pix.3 >> 8) as u8, (prev.3 >> 8) as u8)
            ],
            ColorType::RGB =>  vec![
                sub((pix.0 >> 8) as u8, (prev.0 >> 8) as u8),
                sub((pix.1 >> 8) as u8, (prev.1 >> 8) as u8),
                sub((pix.2 >> 8) as u8, (prev.2 >> 8) as u8)
            ],
            ColorType::NDXA | ColorType::NDX => vec![
                sub((pix.0 & 0xff) as u8, (prev.0 & 0xff) as u8),
            ]
        };
        prev = *pix;
        r
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_up(row: &[RGBA16], above: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();

    res.push(2_u8);

    res.extend(zip(0 .. row.len(), row.iter()).map(|(x, pix)| {
        match color_type {
            ColorType::RGBA16 =>
                vec![
                    sub(((pix.0) >> 8) as u8, (above[x].0 >> 8) as u8),
                    sub(((pix.0) & 0xff) as u8, (above[x].0 & 0xff) as u8),
                    sub(((pix.1) >> 8) as u8, (above[x].1 >> 8) as u8),
                    sub(((pix.1) & 0xff) as u8, (above[x].1 & 0xff) as u8),
                    sub(((pix.2) >> 8) as u8, (above[x].2 >> 8) as u8),
                    sub(((pix.2) & 0xff) as u8, (above[x].2 & 0xff) as u8),
                    sub(((pix.3) >> 8) as u8, (above[x].3 >> 8) as u8),
                    sub(((pix.3) & 0xff) as u8, (above[x].3 & 0xff) as u8)
                ],
            ColorType::RGB16 =>
                vec![
                    sub(((pix.0) >> 8) as u8, (above[x].0 >> 8) as u8),
                    sub(((pix.0) & 0xff) as u8, (above[x].0 & 0xff) as u8),
                    sub(((pix.1) >> 8) as u8, (above[x].1 >> 8) as u8),
                    sub(((pix.1) & 0xff) as u8, (above[x].1 & 0xff) as u8),
                    sub(((pix.2) >> 8) as u8, (above[x].2 >> 8) as u8),
                    sub(((pix.2) & 0xff) as u8, (above[x].2 & 0xff) as u8)
                ],
            ColorType::RGBA =>
                vec![
                    sub(((pix.0) >> 8) as u8, (above[x].0 >> 8) as u8),
                    sub(((pix.1) >> 8) as u8, (above[x].1 >> 8) as u8),
                    sub(((pix.2) >> 8) as u8, (above[x].2 >> 8) as u8),
                    sub(((pix.3) >> 8) as u8, (above[x].3 >> 8) as u8)
                ],
            ColorType::RGB =>
                vec![
                    sub(((pix.0) >> 8) as u8, (above[x].0 >> 8) as u8),
                    sub(((pix.1) >> 8) as u8, (above[x].1 >> 8) as u8),
                    sub(((pix.2) >> 8) as u8, (above[x].2 >> 8) as u8),
                ],
            ColorType::NDXA | ColorType::NDX =>
                vec![
                    sub(pix.0 as u8, above[x].0 as u8)
                ]
        }
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_avg(row: &[RGBA16], above: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    assert_eq!(row.len(), above.len());

    let mut res: Vec<u8> = Vec::new();
    let mut prev: RGBA16 = (0, 0, 0, 0);

    res.push(3_u8);

    res.extend(zip(0 .. row.len(), row.iter()).map(|(x, pix)| {
        let a0h = (((prev.0 >> 8)  + (above[x].0 >> 8)) >> 1) as u8;
        let a1h = (((prev.1 >> 8)  + (above[x].1 >> 8)) >> 1) as u8;
        let a2h = (((prev.2 >> 8)  + (above[x].2 >> 8)) >> 1) as u8;
        let a3h = (((prev.3 >> 8)  + (above[x].3 >> 8)) >> 1) as u8;

        let a0l = ((((prev.0) & 0xff) + (above[x].0 & 0xff)) >> 1) as u8;
        let a1l = ((((prev.1) & 0xff) + (above[x].1 & 0xff)) >> 1) as u8;
        let a2l = ((((prev.2) & 0xff) + (above[x].2 & 0xff)) >> 1) as u8;
        let a3l = ((((prev.3) & 0xff) + (above[x].3 & 0xff)) >> 1) as u8;

        let r = match color_type {
            ColorType::RGBA16 => vec![
                sub((pix.0 >> 8) as u8, a0h),
                sub((pix.0 & 0xff) as u8, a0l),
                sub((pix.1 >> 8) as u8, a1h),
                sub((pix.1 & 0xff) as u8, a1l),
                sub((pix.2 >> 8) as u8, a2h),
                sub((pix.2 & 0xff) as u8, a2l),
                sub((pix.3 >> 8) as u8, a3h),
                sub((pix.3 & 0xff) as u8, a3l),
            ],
            ColorType::RGB16 => vec![
                sub((pix.0 >> 8) as u8, a0h),
                sub((pix.0 & 0xff) as u8, a0l),
                sub((pix.1 >> 8) as u8, a1h),
                sub((pix.1 & 0xff) as u8, a1l),
                sub((pix.2 >> 8) as u8, a2h),
                sub((pix.2 & 0xff) as u8, a2l)
            ],
            ColorType::RGBA => vec![
                sub((pix.0 >> 8) as u8, a0h),
                sub((pix.1 >> 8) as u8, a1h),
                sub((pix.2 >> 8) as u8, a2h),
                sub((pix.3 >> 8) as u8, a3h)
            ],
            ColorType::RGB => vec![
                sub((pix.0 >> 8) as u8, a0h),
                sub((pix.1 >> 8) as u8, a1h),
                sub((pix.2 >> 8) as u8, a2h),
            ],
            ColorType::NDXA | ColorType::NDX => vec![
                sub(pix.0 as u8, a0l)
            ]
        };

        prev = *pix;
        r
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_paeth(row: &[RGBA16], above: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    assert_eq!(row.len(), above.len());

    let mut res: Vec<u8> = Vec::new();
    let mut prev: RGBA16 = (0, 0, 0, 0);

    res.push(4_u8);

    res.extend(zip(0 .. row.len(), row.iter()).map(|(x, pix)| {
        let a0 = prev.0;
        let a1 = prev.1;
        let a2 = prev.2;
        let a3 = prev.3;
        let b0 = above[x].0;
        let b1 = above[x].1;
        let b2 = above[x].2;
        let b3 = above[x].3;
        let c0 = if x == 0 { 0 } else { above[x - 1].0 };
        let c1 = if x == 0 { 0 } else { above[x - 1].1 };
        let c2 = if x == 0 { 0 } else { above[x - 1].2 };
        let c3 = if x == 0 { 0 } else { above[x - 1].3 };

        let a0h = (a0 >> 8) as u8;
        let a1h = (a1 >> 8) as u8;
        let a2h = (a2 >> 8) as u8;
        let a3h = (a3 >> 8) as u8;

        let a0l = (a0 & 0xff) as u8;
        let a1l = (a1 & 0xff) as u8;
        let a2l = (a2 & 0xff) as u8;
        let a3l = (a3 & 0xff) as u8;

        let b0h = (b0 >> 8) as u8;
        let b1h = (b1 >> 8) as u8;
        let b2h = (b2 >> 8) as u8;
        let b3h = (b3 >> 8) as u8;

        let b0l = (b0 & 0xff) as u8;
        let b1l = (b1 & 0xff) as u8;
        let b2l = (b2 & 0xff) as u8;
        let b3l = (b3 & 0xff) as u8;

        let c0h = (c0 >> 8) as u8;
        let c1h = (c1 >> 8) as u8;
        let c2h = (c2 >> 8) as u8;
        let c3h = (c3 >> 8) as u8;

        let c0l = (c0 & 0xff) as u8;
        let c1l = (c1 & 0xff) as u8;
        let c2l = (c2 & 0xff) as u8;
        let c3l = (c3 & 0xff) as u8;

        let r = match color_type {
            ColorType::RGBA16 => vec![
                sub((pix.0 >> 8) as u8, paeth(a0h, b0h, c0h)),
                sub((pix.0 & 0xff) as u8, paeth(a0l, b0l, c0l)),
                sub((pix.1 >> 8) as u8, paeth(a1h, b1h, c1h)),
                sub((pix.1 & 0xff) as u8, paeth(a1l, b1l, c1l)),
                sub((pix.2 >> 8) as u8, paeth(a2h, b2h, c2h)),
                sub((pix.2 & 0xff) as u8, paeth(a2l, b2l, c2l)),
                sub((pix.3 >> 8) as u8, paeth(a3h, b3h, c3h)),
                sub((pix.3 & 0xff) as u8, paeth(a3l, b3l, c3l)),
            ],
            ColorType::RGB16 => vec! [
                sub((pix.0 >> 8) as u8, paeth(a0h, b0h, c0h)),
                sub((pix.0 & 0xff) as u8, paeth(a0l, b0l, c0l)),
                sub((pix.1 >> 8) as u8, paeth(a1h, b1h, c1h)),
                sub((pix.1 & 0xff) as u8, paeth(a1l, b1l, c1l)),
                sub((pix.2 >> 8) as u8, paeth(a2h, b2h, c2h)),
                sub((pix.2 & 0xff) as u8, paeth(a2l, b2l, c2l)),
            ],
            ColorType::RGBA => vec![
                sub((pix.0 >> 8) as u8, paeth(a0h, b0h, c0h)),
                sub((pix.1 >> 8) as u8, paeth(a1h, b1h, c1h)),
                sub((pix.2 >> 8) as u8, paeth(a2h, b2h, c2h)),
                sub((pix.3 >> 8) as u8, paeth(a3h, b3h, c3h)),
            ],
            ColorType::RGB => vec! [
                sub((pix.0 >> 8) as u8, paeth(a0h, b0h, c0h)),
                sub((pix.1 >> 8) as u8, paeth(a1h, b1h, c1h)),
                sub((pix.2 >> 8) as u8, paeth(a2h, b2h, c2h)),
            ],
            ColorType::NDXA | ColorType::NDX => vec![
                sub((pix.0 & 0xff) as u8, paeth(a0l, b0l, c0l))
            ]
        };

        prev = *pix;
        r
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_rev(left: u8, up: u8, corner: u8, est: PkEst) -> u8 {
    match est {
        PkEst::None => 0,
        PkEst::Sub => left,
        PkEst::Up => up,
        PkEst::Avg => ((left as u16 + up as u16) >> 1) as u8,
        PkEst::Paeth => paeth(left, up, corner)
    }
}

fn pk_size(payload: &[u8]) -> usize {
    compress(payload, CompressionLevel::Lvl1).len() // NOTE quick estimation
    //let window = &payload[
        ////(payload.len() - (1 << 15)).clamp(0, payload.len())
     //..];

    //let mut e = ZlibEncoder::new(Vec::new(), Compression::fast());

    //match e.write_all(payload) {
        //Ok(_) => (),
        //Err(_) => panic!("compress")
    //};

    ////e.flush().unwrap();
    ////return e.total_out() as usize;

    //match e.finish() {
        //Ok(d) => d.len(),
        //Err(_) => panic!("zlib flush")
    //}
}

async fn estimate_worker(est: PkEst, so_far: Vec<u8>, payload: Vec<u8>) -> (PkEst, usize, Vec<u8>) {
    async fn doit(_so_far: Vec<u8>, payload: Vec<u8>) -> usize {
        //let mut step: Vec<u8> = so_far;
        //step.extend(payload);
        pk_size(&payload[..])
    }

    (est, spawn(doit(so_far, payload.clone())).await, payload)
}

fn elect_best(preds: &[(PkEst, Pred)], so_far: Vec<u8>, line: &[RGBA16], above: &[RGBA16], color_type: ColorType) ->
    (PkEst, Vec<u8>) {

    let tasks = preds.iter().map(|(id, func)| {
        estimate_worker(*id, so_far.clone(), func(line, above, color_type))
    });

    let output = futures_executor::block_on(join_all(tasks));

    let mut best = PkEst::None;
    let mut bsize = output[0].1;
    let mut bpayload: Vec<u8> = output[0].2.clone();

    output.iter().for_each(|(key, size, payload)| {
        if bsize > *size {
            best = *key;
            bsize = *size;
            bpayload = payload.clone();
        }
    });

    (best, bpayload)
}

fn prepare_frames(image_data: &ImageData) -> Vec<Vec<Vec<RGBA16>>> {
    match image_data {
        ImageData::RGBA16(rgba) =>
            rgba.iter().map(|frame| -> Vec<Vec<RGBA16>> {
                frame.iter().map(|line| -> Vec<RGBA16> {
                    line.iter().map(|pix| -> RGBA16 {
                        (
                            pix.0,
                            pix.1,
                            pix.2,
                            pix.3
                        )
                    }).collect()
                }).collect()
            }).collect(),
        ImageData::RGB16(rgb) =>
            rgb.iter().map(|frame| -> Vec<Vec<RGBA16>> {
                frame.iter().map(|line| -> Vec<RGBA16> {
                    line.iter().map(|pix| -> RGBA16 {
                        (
                            pix.0,
                            pix.1,
                            pix.2,
                            0xffff_u16
                        )
                    }).collect()
                }).collect()
            }).collect(),
        ImageData::RGBA(rgba) =>
            rgba.iter().map(|frame| -> Vec<Vec<RGBA16>> {
                frame.iter().map(|line| -> Vec<RGBA16> {
                    line.iter().map(|pix| -> RGBA16 {
                        (
                            (pix.0 as u16) << 8,
                            (pix.1 as u16) << 8,
                            (pix.2 as u16) << 8,
                            (pix.3 as u16) << 8
                        )
                    }).collect()
                }).collect()
            }).collect(),
        ImageData::RGB(rgb) =>
            rgb.iter().map(|frame| -> Vec<Vec<RGBA16>> {
                frame.iter().map(|line| -> Vec<RGBA16> {
                    line.iter().map(|pix| -> RGBA16 {
                        (
                            (pix.0 as u16) << 8,
                            (pix.1 as u16) << 8,
                            (pix.2 as u16) << 8,
                            0xffff_u16
                        )
                    }).collect()
                }).collect()
            }).collect(),
        ImageData::NDXA(ndx, _) | ImageData::NDX(ndx, _) =>
            ndx.iter().map(|frame| -> Vec<Vec<RGBA16>> {
                frame.iter().map(|line| -> Vec<RGBA16> {
                    line.iter().map(|pix| -> RGBA16 {
                        (
                            *pix as u16,
                            0,
                            0,
                            0
                        )
                    }).collect()
                }).collect()
            }).collect()
    }
}

fn gen_palette(image_data: &ImageData) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();

    match image_data {
        ImageData::NDXA(_, pal) => {
            let mut plte: Vec<u8> = b"PLTE".to_vec();

            plte.extend(pal.iter().map(|rgba: &RGBA| {
                vec![
                    rgba.0,
                    rgba.1,
                    rgba.2
                ]
            }).collect::<Vec<Vec<u8>>>().iter().flatten());

            res.extend(png_chunk(&plte));

            let mut trns: Vec<u8> = b"tRNS".to_vec();

            trns.extend(pal.iter().map(|rgba: &RGBA| {
                rgba.3
            }).collect::<Vec<u8>>());

            res.extend(png_chunk(&trns));
        },
        ImageData::NDX(_, pal) => {
            let mut plte: Vec<u8> = b"PLTE".to_vec();

            plte.extend(pal.iter().map(|rgb: &RGB| {
                vec![
                    rgb.0,
                    rgb.1,
                    rgb.2
                ]
            }).collect::<Vec<Vec<u8>>>().iter().flatten());

            res.extend(png_chunk(&plte));
        },
        _ => ()
    }
    res
}

#[derive(Default)]
struct Stats {
    n_none: usize,
    n_sub: usize,
    n_up: usize,
    n_avg: usize,
    n_paeth: usize
}

#[allow(clippy::too_many_arguments)]
fn emit_frame(color_type: ColorType, progress: Option<APNGProgress>,
    stats: &mut Stats, ndx: u32, forced: Option<PkEst>, seq: &mut u32, frames: &Vec<Vec<Vec<RGBA16>>>,
    adam_7: bool) -> Vec<u8> {

    let preds_first: Vec<(PkEst, Pred)> = vec![
        (PkEst::None, png_none),
        (PkEst::Sub, png_sub)
    ];

    let preds_next: Vec<(PkEst, Pred)> = vec![
        (PkEst::None, png_none),
        (PkEst::Sub, png_sub),
        (PkEst::Up, png_up),
        (PkEst::Avg, png_avg),
        (PkEst::Paeth, png_paeth)
    ];

    let mut payload: Vec<u8> = Vec::new();
    //let mut res: Vec<u8> = Vec::new();
    let mut above: Vec<RGBA16> = vec![];

    let mut fdat: Vec<u8> = b"fdAT".to_vec(); // vec![b'f', b'd', b'A', b'T'];

    if ndx == 0 {
        fdat = b"IDAT".to_vec(); // vec![b'I', b'D', b'A', b'T'];
        *seq += 1;
    }
    else {
        *seq += 1;
        fdat.extend((*seq - 2).to_be_bytes());
    }

    //let frame = if adam_7 {
        //let mut data: Vec<Vec<RGBA16>> = Vec::new();
        //let mut ox: usize = 0;
        ////let mut oy: usize = 0;

        //let mut line: Vec<RGBA16> = Vec::new();

        //(1 ..= ADAM_7_SZ).for_each(|a| {
            //(0 .. frames[0].len()).for_each(|y| {
                //(0 .. frames[0][0].len()).for_each(|x| {
                    //let ax = x % ADAM_7_SZ;
                    //let ay = y % ADAM_7_SZ;

                    //if ADAM_7[ax + ADAM_7_SZ * ay] == a {
                        //line.push(frames[ndx as usize][y][x]);
                        //ox += 1;
                        //if ox == frames[ndx as usize].len() {
                            //ox = 0;
                            //data.push(line.clone());
                            //line = vec![];
                        //}
                    //}
                //})
            //});
        //});

        //println!("{} {}", data.len(), data[0].len());

        //data
    //}
    //else {
        //frames[ndx as usize].clone()
    //};


    let frame = &frames[ndx as usize];
    let mut first = true;

    let a_lo = if adam_7 { 1 } else { 0 };
    let a_hi = if adam_7 { 7 } else { 0 };

    (a_lo ..= a_hi).for_each(|a| {
        first = true;
        above = vec![];

        zip(0 .. frame.len(), frame.iter()).for_each(|(y, row_raw)| {
            if let Some(p) = progress {
                p(y, frame.len(), &"lines".to_string());
            }

            let mut so_far: Vec<u8> = vec![];

            let line = if adam_7 && frames.len() == 1 {
                let ay = y % ADAM_7_SZ;
                let mut row: Vec<RGBA16> = Vec::new();

                zip(0 .. row_raw.len(), row_raw).for_each(|(x, pix)| {
                    let ax = x % ADAM_7_SZ;

                    if ADAM_7[ax + ADAM_7_SZ * ay] == a {
                        row.push(*pix)
                    }
                });

                row
            }
            else {
                let line = if ndx == 0 {
                    row_raw.clone()
                }
                else {
                    zip(row_raw.iter(), frames[ndx as usize - 1][y].iter()).map(|(cur, prev)| {
                        if (color_type == ColorType::RGBA || color_type == ColorType::RGBA16) && *cur == *prev {
                            (0, 0, 0, 0)
                        }
                        else {
                            *cur
                        }
                    }).collect::<Vec<RGBA16>>()
                };
                line
            };

            if line.is_empty() {
                return
            }

            if first {
                match forced {
                    Some(PkEst::None) => {
                        let p_none = png_none(&line[..], &[], color_type);
                        payload.extend(&p_none);
                        stats.n_none += 1;
                    },
                    Some(_) => {
                        let p_sub = png_sub(&line[..], &[], color_type);
                        payload.extend(&p_sub);
                        stats.n_sub += 1;
                    },
                    None => { // NOTE elect
                        let (best, bpayload) = elect_best(&preds_first, so_far.clone(), &line, &Vec::new(), color_type);

                        so_far.extend(&bpayload);
                        payload.extend(bpayload);

                        match best {
                            PkEst::None => stats.n_none += 1,
                            PkEst::Sub => stats.n_sub += 1,
                            _ => panic!("pred elect @ one line: failure")
                        }
                    }
                }
                first = false;
            }
            else {
                match forced {
                    Some(PkEst::None) => {
                        stats.n_none += 1;
                        payload.extend(png_none(&line[..], &above[..], color_type));
                    },
                    Some(PkEst::Sub) => {
                        stats.n_sub += 1;
                        payload.extend(png_sub(&line[..], &above[..], color_type));
                    },
                    Some(PkEst::Up) => {
                        stats.n_up += 1;
                        payload.extend(png_up(&line[..], &above[..], color_type));
                    },
                    Some(PkEst::Avg) => {
                        stats.n_avg += 1;
                        payload.extend(png_avg(&line[..], &above[..], color_type));
                    },
                    Some(PkEst::Paeth) => {
                        stats.n_paeth += 1;
                        payload.extend(png_paeth(&line[..], &above[..], color_type));
                    },
                    None => { // NOTE elect
                        let (best, bpayload) = elect_best(&preds_next, so_far.clone(), &line, &above, color_type);

                        so_far.extend(&bpayload);
                        payload.extend(bpayload);

                        match best {
                            PkEst::None => stats.n_none += 1,
                            PkEst::Sub => stats.n_sub += 1,
                            PkEst::Up => stats.n_up += 1,
                            PkEst::Avg => stats.n_avg += 1,
                            PkEst::Paeth => stats.n_paeth += 1
                        }
                    }
                }
            }

            above = line;
        });
    });

    let mut e = ZlibEncoder::new(Vec::new(), Compression::best());

    e.write_all(&payload).expect("compress");
    let c = e.finish().expect("zlib flush");

    fdat.extend(&c);

    png_chunk(&fdat)
}

fn apng_frames(image_data: &ImageData, forced: Option<PkEst>, progress: Option<APNGProgress>, mut adam_7: bool)
    -> Result<Vec<u8>, String> {
    let color_type = match image_data {
        ImageData::RGBA16(_) => ColorType::RGBA16,
        ImageData::RGB16(_) => ColorType::RGB16,
        ImageData::RGBA(_) => ColorType::RGBA,
        ImageData::RGB(_) => ColorType::RGB,
        ImageData::NDXA(_, _) => ColorType::NDXA,
        ImageData::NDX(_, _) => ColorType::NDX
    };

    let frames = prepare_frames(image_data);

    let mut res: Vec<u8> = vec![];

    res.extend(b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a");

    let mut ihdr: Vec<u8> = b"IHDR".to_vec(); // vec![b'I', b'H', b'D', b'R'];

    let width = frames[0][0].len();
    let height = frames[0].len();

    ihdr.extend((width as u32).to_be_bytes());
    ihdr.extend((height as u32).to_be_bytes());

    let color_byte = match color_type {
        ColorType::RGBA16 => b'\x06',
        ColorType::RGB16 => b'\x02',
        ColorType::RGBA => b'\x06',
        ColorType::RGB => b'\x02',
        ColorType::NDX => b'\x03',
        ColorType::NDXA => b'\x03',
    };

    let bpp = match color_type {
        ColorType::RGBA16 => b'\x10',
        ColorType::RGB16 => b'\x10',
        ColorType::RGBA => b'\x08',
        ColorType::RGB => b'\x08',
        ColorType::NDX => b'\x08',
        ColorType::NDXA => b'\x08',
    };

    ihdr.append(&mut vec![bpp, color_byte, b'\x00', b'\x00', if adam_7 { b'\x01' } else { b'\x00' }]);

    res.extend(png_chunk(&ihdr));

    res.extend(gen_palette(image_data));

    let mut text: Vec<u8> = b"tEXt".to_vec();
    text.extend(b"Comment\x00created by nobody / SQ6KBQ".to_vec());
    res.extend(png_chunk(&text));

    if frames.len() > 1 {
        let mut actl: Vec<u8> = b"acTL".to_vec(); // vec![b'a', b'c', b'T', b'L'];

        actl.extend((frames.len() as u32).to_be_bytes());
        actl.extend((0_u32).to_be_bytes());

        res.extend(png_chunk(&actl));
        adam_7 = false;
    }

    let mut seq = 1_u32;

    let mut stats = Stats::default();

    zip(0 .. frames.len() as u32, frames.iter()).for_each(|(ndx, frame)| {
        if frames.len() > 1 && progress.is_some() {
            if let Some(p) = progress {
                p(ndx as usize, frames.len(), &"frames".to_string());
            }
        }

        if frames.len() > 1 {
            let mut fctl: Vec<u8> = b"fcTL".to_vec(); // vec![b'f', b'c', b'T', b'L'];

            if ndx > 0 {
                fctl.extend((seq - 1).to_be_bytes());
                seq += 1;
            }
            else {
                fctl.extend(ndx.to_be_bytes());
            }

            fctl.extend((frame[0].len() as u32).to_be_bytes());
            fctl.extend((frame.len() as u32).to_be_bytes());

            fctl.extend((0_u32).to_be_bytes());
            fctl.extend((0_u32).to_be_bytes());

            fctl.extend((1_u16).to_be_bytes());// delay num
            fctl.extend((100_u16).to_be_bytes());// delay den

            fctl.extend((0_u8).to_be_bytes());// dispose, 0 - copy
            fctl.extend((1_u8).to_be_bytes());// blend, 0 - source, 1 - over

            res.extend(png_chunk(&fctl));
        }

        res.extend(emit_frame(
            color_type,
            if frames.len() == 1 {
                progress
            }
            else {
                None
            },
            &mut stats,
            ndx,
            forced,
            &mut seq,
            &frames,
            adam_7
        ));
    });

    let iend: Vec<u8> = b"IEND".to_vec(); // vec![b'I', b'E', b'N', b'D'];

    res.extend(png_chunk(&iend));

    if let Some(p) = progress {
        if frames.len() > 1 {
            p(frames.len(), frames.len(),
                &format!("n:{} / s:{} / u:{} / a:{} / p:{}", stats.n_none, stats.n_sub, stats.n_up, stats.n_avg, stats.n_paeth));
        }
        else {
            p(frames[0].len(), frames[0].len(),
                                &format!("n:{} / s:{} / u:{} / a:{} / p:{}", stats.n_none, stats.n_sub, stats.n_up, stats.n_avg, stats.n_paeth));
        }
    }

    Ok(res)
}

pub fn write_apng(fname: &String, image_data: &ImageData, forced: Option<PkEst>,
    progress: Option<APNGProgress>, adam_7: bool) -> Result<(), String> {
    if let Ok(mut f) = File::create(fname) {
        if f.write_all(&apng_frames(image_data, forced, progress, adam_7)?).is_err() {
            Err("write error".to_string())
        }
        else {
            Ok(())
        }
    }
    else {
        Err("open error".to_string())
    }
}

fn get_chunk(inp: &[u8]) -> Result<(String, &[u8]), String> {
    let length = match inp[0 .. 4].try_into() {
        Ok(l) => u32::from_be_bytes(l) as usize,
        Err(_) => return Err("cannot unpack length".to_string())
    };

    let code = match std::str::from_utf8(&inp[4 .. 8]) {
        Ok(c) => c.to_string(),
        Err(_) => return Err("cannot unpack chunk code".to_string())
    };

    let mut crc = Hasher::new();
    crc.update(&inp[4 .. 8 + length]);
    let crc_got = crc.finalize().to_be_bytes();

    if crc_got != inp[8 + length .. 12 + length] {
        return Err("bad crc".to_string())
    }

    Ok((code, &inp[8 .. 8 + length]))
}

fn unpack_idat(width: usize, height: usize, raw: &[u8], color_type: ColorType, pal: &Vec<RGBA>) -> Result<(Vec<Vec<RGBA16>>, ImageData), String> {
    let mut res: Vec<Vec<RGBA16>> = Vec::new();

    let mut raw_rgba16: Vec<Vec<RGBA16>> = Vec::new();
    let mut raw_rgb16: Vec<Vec<RGB16>> = Vec::new();
    let mut raw_rgba: Vec<Vec<RGBA>> = Vec::new();
    let mut raw_rgb: Vec<Vec<RGB>> = Vec::new();
    let mut raw_ndx: Vec<Vec<u8>> = Vec::new();

    let slice_len = match color_type {
        ColorType::RGBA16 => width * 4 * 2 + 1,
        ColorType::RGB16 => width * 3 * 2 + 1,
        ColorType::RGBA => width * 4 + 1,
        ColorType::RGB => width * 3 + 1,
        ColorType::NDXA | ColorType::NDX => width + 1
    };

    let mut above: Vec<RGBA16> = vec![(0, 0, 0, 0); width];
    let mut ndx_above: Vec<u8> = vec![0; width];
    let mut ndx_line: Vec<u8> = Vec::new();

    let status = (0 .. height).map(|y| -> Result<(), String> {
        let offs = match color_type {
            ColorType::RGBA16 => (width * 4 * 2 + 1) * y,
            ColorType::RGB16 => (width * 3 * 2 + 1) * y,
            ColorType::RGBA => (width * 4 + 1) * y,
            ColorType::RGB => (width * 3 + 1) * y,
            ColorType::NDXA | ColorType::NDX => (width + 1) * y
        };
        let slice = &raw[offs .. offs + slice_len];
        let mode = slice[0];
        let mut line: Vec<RGBA16> = Vec::new();

        let est = match mode {
            0 => PkEst::None,
            1 => PkEst::Sub,
            2 => PkEst::Up,
            3 => PkEst::Avg,
            4 => PkEst::Paeth,
            e => return Err(format!("bad est: {e}"))
        };

        match color_type {
            ColorType::RGBA16 => {
                let mut rgba16_line: Vec<RGBA16> = Vec::new();
                (0 .. width * 4 * 2).step_by(8).for_each(|ox| {
                    let x = ox >> 3;
                    let rh = add(slice[ox + 1], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].0 >> 8) as u8},
                        (above[x].0 >> 8) as u8,
                        if x < 1 { 0 } else { (above[x - 1].0 >> 8) as u8 },
                        est
                    ));
                    let rl = add(slice[ox + 2], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].0 & 0xff) as u8},
                        (above[x].0 & 0xff ) as u8,
                        if x < 1 { 0 } else { (above[x - 1].0 & 0xff) as u8 },
                        est
                    ));

                    let gh = add(slice[ox + 3], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].1 >> 8) as u8},
                        (above[x].1 >> 8) as u8,
                        if x < 1 { 0 } else { (above[x - 1].1 >> 8) as u8 },
                        est
                    ));
                    let gl = add(slice[ox + 4], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].1 & 0xff) as u8},
                        (above[x].1 & 0xff ) as u8,
                        if x < 1 { 0 } else { (above[x - 1].1 & 0xff) as u8 },
                        est
                    ));

                    let bh = add(slice[ox + 5], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].2 >> 8) as u8},
                        (above[x].2 >> 8) as u8,
                        if x < 1 { 0 } else { (above[x - 1].2 >> 8) as u8 },
                        est
                    ));
                    let bl = add(slice[ox + 6], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].2 & 0xff) as u8},
                        (above[x].2 & 0xff ) as u8,
                        if x < 1 { 0 } else { (above[x - 1].2 & 0xff) as u8 },
                        est
                    ));

                    let ah = add(slice[ox + 7], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].3 >> 8) as u8},
                        (above[x].3 >> 8) as u8,
                        if x < 1 { 0 } else { (above[x - 1].3 >> 8) as u8 },
                        est
                    ));
                    let al = add(slice[ox + 8], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].3 & 0xff) as u8},
                        (above[x].3 & 0xff ) as u8,
                        if x < 1 { 0 } else { (above[x - 1].3 & 0xff) as u8 },
                        est
                    ));

                    let pix: RGBA16 = (
                        ((rh as u16) << 8) + (rl as u16),
                        ((gh as u16) << 8) + (gl as u16),
                        ((bh as u16) << 8) + (bl as u16),
                        ((ah as u16) << 8) + (al as u16)
                    );

                    line.push(pix);
                    rgba16_line.push(pix);
                });
                raw_rgba16.push(rgba16_line);
            },
            ColorType::RGB16 => {
                let mut rgb16_line: Vec<RGB16> = Vec::new();
                (0 .. width * 3 * 2).step_by(6).for_each(|ox| {
                    let x = ox / (3 * 2);
                    let rh = add(slice[ox + 1], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].0 >> 8) as u8},
                        (above[x].0 >> 8) as u8,
                        if x < 1 { 0 } else { (above[x - 1].0 >> 8) as u8 },
                        est
                    ));
                    let rl = add(slice[ox + 2], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].0 & 0xff) as u8},
                        (above[x].0 & 0xff ) as u8,
                        if x < 1 { 0 } else { (above[x - 1].0 & 0xff) as u8 },
                        est
                    ));

                    let gh = add(slice[ox + 3], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].1 >> 8) as u8},
                        (above[x].1 >> 8) as u8,
                        if x < 1 { 0 } else { (above[x - 1].1 >> 8) as u8 },
                        est
                    ));
                    let gl = add(slice[ox + 4], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].1 & 0xff) as u8},
                        (above[x].1 & 0xff ) as u8,
                        if x < 1 { 0 } else { (above[x - 1].1 & 0xff) as u8 },
                        est
                    ));

                    let bh = add(slice[ox + 5], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].2 >> 8) as u8},
                        (above[x].2 >> 8) as u8,
                        if x < 1 { 0 } else { (above[x - 1].2 >> 8) as u8 },
                        est
                    ));
                    let bl = add(slice[ox + 6], png_rev(
                        if x < 1 { 0 } else { (line[x - 1].2 & 0xff) as u8},
                        (above[x].2 & 0xff ) as u8,
                        if x < 1 { 0 } else { (above[x - 1].2 & 0xff) as u8 },
                        est
                    ));

                    let pix: RGBA16 = (
                        ((rh as u16) << 8) + (rl as u16),
                        ((gh as u16) << 8) + (gl as u16),
                        ((bh as u16) << 8) + (bl as u16),
                        0xffff_u16
                    );

                    line.push(pix);
                    rgb16_line.push((
                        pix.0,
                        pix.1,
                        pix.2
                    ));
                });
                raw_rgb16.push(rgb16_line);
            },
            ColorType::RGBA => {
                let mut rgba_line: Vec<RGBA> = Vec::new();
                (0 .. width * 4).step_by(4).for_each(|ox| {
                    let x = ox >> 2;
                    let pix: RGBA = (
                        add(slice[ox + 1], png_rev(
                            if x < 1 { 0 } else { (line[x - 1].0 >> 8) as u8},
                            (above[x].0 >> 8) as u8,
                            if x < 1 { 0 } else { (above[x - 1].0 >> 8) as u8 },
                            est
                        )),
                        add(slice[ox + 2], png_rev(
                            if x < 1 { 0 } else { (line[x - 1].1 >> 8) as u8},
                            (above[x].1 >> 8) as u8,
                            if x < 1 { 0 } else { (above[x - 1].1 >> 8) as u8 },
                            est
                        )),
                        add(slice[ox + 3], png_rev(
                            if x < 1 { 0 } else { (line[x - 1].2 >> 8) as u8},
                            (above[x].2 >> 8) as u8,
                            if x < 1 { 0 } else { (above[x - 1].2 >> 8) as u8 },
                            est
                        )),
                        add(slice[ox + 4], png_rev(
                            if x < 1 { 0 } else { (line[x - 1].3 >> 8) as u8},
                            (above[x].3 >> 8) as u8,
                            if x < 1 { 0 } else { (above[x - 1].3 >> 8) as u8 },
                            est
                        ))
                    );
                    line.push((
                        pix.0 as u16 | (pix.0 as u16) << 8,
                        pix.1 as u16 | (pix.1 as u16) << 8,
                        pix.2 as u16 | (pix.2 as u16) << 8,
                        pix.3 as u16 | (pix.3 as u16) << 8
                    ));
                    rgba_line.push(pix);
                });
                raw_rgba.push(rgba_line);
            },
            ColorType::RGB => {
                let mut rgb_line: Vec<RGB> = Vec::new();

                (0 .. width * 3).step_by(3).for_each(|ox| {
                    let x = ox / 3;
                    let pix: RGBA = (
                        add(slice[ox + 1], png_rev(
                            if x < 1 { 0 } else { (line[x - 1].0 >> 8) as u8},
                            (above[x].0 >> 8) as u8,
                            if x < 1 { 0 } else { (above[x - 1].0 >> 8) as u8 },
                            est
                        )),
                        add(slice[ox + 2], png_rev(
                            if x < 1 { 0 } else { (line[x - 1].1 >> 8) as u8},
                            (above[x].1 >> 8) as u8,
                            if x < 1 { 0 } else { (above[x - 1].1 >> 8) as u8 },
                            est
                        )),
                        add(slice[ox + 3], png_rev(
                            if x < 1 { 0 } else { (line[x - 1].2 >> 8) as u8},
                            (above[x].2 >> 8) as u8,
                            if x < 1 { 0 } else { (above[x - 1].2 >> 8) as u8 },
                            est
                        )),
                        0xff_u8
                    );
                    rgb_line.push((
                        pix.0,
                        pix.1,
                        pix.2
                    ));
                    line.push((
                        pix.0 as u16 | (pix.0 as u16) << 8,
                        pix.1 as u16 | (pix.1 as u16) << 8,
                        pix.2 as u16 | (pix.2 as u16) << 8,
                        0xffff_u16
                    ));
                });
                raw_rgb.push(rgb_line);
            },
            ColorType::NDXA | ColorType::NDX => {
                ndx_line = Vec::new();
                //let mut ndx_line: Vec<u8> = Vec::new();
                let pal_status = (0 .. width).map(|ox| -> Result<(), String> {
                    let x = ox;
                    let ndx = add(slice[ox + 1], png_rev(
                            if x < 1 { 0 } else { ndx_line[x - 1]},
                            ndx_above[x],
                            if x < 1 { 0 } else { ndx_above[x - 1]},
                            est
                        )
                    );
                    if ndx as usize >= pal.len() {
                        return Err("palette overflow".to_string());
                    }
                    let pix: RGBA16 = (
                        (pal[ndx as usize].0 as u16) << 8,
                        (pal[ndx as usize].1 as u16) << 8,
                        (pal[ndx as usize].2 as u16) << 8,
                        (pal[ndx as usize].3 as u16) << 8
                    );
                    // HERE - push shifted to left
                    line.push((
                        pix.0,
                        pix.1,
                        pix.2,
                        if ColorType::NDX == color_type {
                            0xffff_u16
                        }
                        else {
                            pix.3
                        }
                    ));
                    ndx_line.push(ndx);
                    Ok(())
                }).find(|x| x.is_err());
                if let Some(e) = pal_status {
                    return e;
                }
                raw_ndx.push(ndx_line.clone());
            }
        };

        above = line.clone();
        res.push(line);

        ndx_above = ndx_line.clone();

        Ok(())
    }).find(|x| x.is_err());

    let raw_data = match color_type {
        ColorType::RGBA16 => ImageData::RGBA16(vec![raw_rgba16]),
        ColorType::RGB16 => ImageData::RGB16(vec![raw_rgb16]),
        ColorType::RGBA => ImageData::RGBA(vec![raw_rgba]),
        ColorType::RGB => ImageData::RGB(vec![raw_rgb]),
        ColorType::NDXA => ImageData::NDXA(vec![raw_ndx], pal.clone()),
        ColorType::NDX => {
            let pal_rgb: Vec<RGB> = pal.iter().map(|pal| {
                (
                    pal.0,
                    pal.1,
                    pal.2
                )
            }).collect();
            ImageData::NDX(vec![raw_ndx], pal_rgb)
        }
    };

    match status {
        Some(Err(e)) => Err(e),
        Some(Ok(_)) => Err("internal unpack_idat error".to_string()),
        None => Ok((res, raw_data))
    }
}

pub fn read_png(fname: &String) -> Result<Image, String> {
    let mut input = match File::open(fname) {
        Ok(f) => f,
        Err(_) => return Err("cannot open file".to_string())
    };

    let input_len = match fs::metadata(fname) {
        Ok(meta) => meta.len() as usize,
        Err(_) => return Err("cannot get metadata".to_string())
    };

    let mut buf = vec![0_u8; input_len];

    match input.read_exact(&mut buf) {
        Ok(_) => (),
        Err(_) => return Err("cannot read file".to_string())
    };

    let header: Vec<u8> = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a".to_vec();

    if header[..] != buf[.. header.len()] {
        return Err("header broken".to_string())
    }

    let mut offs = header.len();

    let mut width: usize = 0;
    let mut height: usize = 0;
    let mut data: Vec<Vec<RGBA16>> = Vec::new();
    let mut color_type: ColorType = ColorType::RGBA;

    let mut pal: Vec<RGBA> = vec![(0, 0, 0, 0xff); 256];

    let mut meta: HashMap<String, String> = HashMap::new();

    let mut unpacked: Vec<u8> = Vec::new();
    let mut prev_idat = false;

    let mut raw: Option<ImageData> = None;

    loop {
        let chunk = get_chunk(&buf[offs ..])?;
        println!("{} / {}", chunk.0, chunk.1.len());

        offs += chunk.1.len() + 12_usize;// len, crc

        if chunk.0 == "IHDR" {
            width = match chunk.1[0 .. 4].try_into() {
                Ok(w) => u32::from_be_bytes(w) as usize,
                Err(_) => return Err("cannot extract width".to_string())
            };

            height = match chunk.1[4 .. 8].try_into() {
                Ok(h) => u32::from_be_bytes(h) as usize,
                Err(_) => return Err("cannot extract height".to_string())
            };

            let depth = chunk.1[8];
            let color = chunk.1[9];
            let pack = chunk.1[10]; // TODO expect 0
            let filter = chunk.1[11];// TODO expect 0
            let ilace = chunk.1[12];// NOTE 1 is Adam7

            if depth != 8 && depth != 16 {
                return Err("unsupported depth".to_string());
            }

            match color {
                6 => if depth == 8 { color_type = ColorType::RGBA } else { color_type = ColorType::RGBA16 },
                2 => if depth == 8 { color_type = ColorType::RGB } else { color_type = ColorType::RGB16 },
                3 => color_type = ColorType::NDX, // NOTE upgrade to NDXA av
                c => return Err(format!("color type {c} not supported"))
            }

            if pack != 0 {
                return Err(format!("unsupported compression {pack}"));
            }

            if filter != 0 {
                return Err(format!("unsupported filter {filter}"));
            }

            if ilace != 0 {
                return Err("Adam7 not supported".to_string());
            }

            println!("{width} x {height}");
        }

        if prev_idat && chunk.0 != "IDAT" {
            let mut deco = ZlibDecoder::new(&unpacked[..]);
            let mut unpacked: Vec<u8> = Vec::new();

            let exp = match color_type {
                ColorType::RGBA16 => (width * 4 * 2 + 1) * height,
                ColorType::RGB16 => (width * 3 * 2 + 1) * height,
                ColorType::RGBA => (width * 4 + 1) * height,
                ColorType::RGB => (width * 3 + 1) * height,
                ColorType::NDXA | ColorType::NDX => (width + 1) * height
            };

            if let Err(e) = deco.read_to_end(&mut unpacked) {
                return Err(format!("zlib IDAT: {e:?}"))
            }

            if unpacked.len() != exp {
                return Err(format!("compression error: exp {exp} got {}", unpacked.len()))
            }

            let (d, r) = unpack_idat(width, height, &unpacked[..], color_type, &pal)?;

            data = d;
            raw = Some(r);
        }

        if chunk.0 == "IDAT" {
            unpacked.extend(chunk.1);
            prev_idat = true;
        }

        if chunk.0 == "PLTE" && (ColorType::NDXA == color_type || ColorType::NDX == color_type) {
            let ps = chunk.1.len() / 3;

            (0 .. ps).for_each(|pndx| {
                pal[pndx].0 = chunk.1[pndx * 3];
                pal[pndx].1 = chunk.1[pndx * 3 + 1];
                pal[pndx].2 = chunk.1[pndx * 3 + 2];
            });
            // NOTE just ignore
        }

        if chunk.0 == "tRNS" {
            if let ColorType::NDX = color_type {
                (0 .. chunk.1.len()).for_each(|pndx| {
                    pal[pndx].3 = chunk.1[pndx];
                });

                color_type = ColorType::NDXA; // NOTE upgrade
            }
            // NOTE just ignore
        }

        if chunk.0 == "tEXt" {
            let mut key: Vec<u8> = Vec::new();
            let mut val: Vec<u8> = Vec::new();
            let mut key_mode = true;

            chunk.1.iter().for_each(|byte| {
                if *byte == 0 && key_mode {
                    key_mode = false
                }
                else if key_mode {
                    key.push(*byte)
                }
                else {
                    val.push(*byte)
                }
            });

            meta.insert(match String::from_utf8(key) {
                Ok(s) => s,
                Err(e) => return Err(e.to_string())
            }, match String::from_utf8(val) {
                Ok(s) => s,
                Err(e) => return Err(e.to_string())
            });
        }

        if chunk.0 == "zTXt" {
            let mut key: Vec<u8> = Vec::new();
            let mut val: Vec<u8> = Vec::new();
            let mut key_mode = true;

            chunk.1.iter().for_each(|byte| {
                if *byte == 0 && key_mode {
                    key_mode = false
                }
                else if key_mode {
                    key.push(*byte)
                }
                else {
                    val.push(*byte)
                }

            });

            if val[0] != 0 {
                return Err(format!("bad zTXt compression: {}", val[0]))
            }

            let mut deco = ZlibDecoder::new(&val[1 ..]);
            let mut unpacked: Vec<u8> = Vec::new();

            if let Err(e) = deco.read_to_end(&mut unpacked) {
                return Err(format!("zlib META: {e:?}"))
            }

            meta.insert(match String::from_utf8(key) {
                Ok(s) => s,
                Err(e) => return Err(e.to_string())
            }, String::from_utf8(unpacked).unwrap());
        }

        if chunk.0 == "zTXt" {
        }

        if chunk.0 == "IEND" {
            break
        }
    }

    Ok(Image {
        width,
        height,
        color_type,
        data,
        meta,
        raw: raw.unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_rgba() -> (Vec<Vec<RGBA16>>, Vec<Vec<RGBA>>) {
        let w = 256_usize;
        let h = 196_usize;

        let mut res: Vec<Vec<RGBA>> = Vec::new();
        let mut orig: Vec<Vec<RGBA16>> = Vec::new();

        (0 .. h).for_each(|y| {
            let mut line: Vec<RGBA> = Vec::new();
            let mut oline: Vec<RGBA16> = Vec::new();

            (0 .. w).for_each(|x| {
                let pix: RGBA = (
                    (x * y) as u8,
                    ((255 - x) * y) as u8,
                    (x * (255 - y)) as u8,
                    ((255 - x) * (255 - y)) as u8
                );

                line.push(pix);

                oline.push((
                    pix.0 as u16 | (pix.0 as u16) << 8,
                    pix.1 as u16 | (pix.1 as u16) << 8,
                    pix.2 as u16 | (pix.2 as u16) << 8,
                    pix.3 as u16 | (pix.3 as u16) << 8
                ));
            });

            res.push(line);
            orig.push(oline);
        });

        (orig, res)
    }

    fn image_rgba_16() -> (Vec<Vec<RGBA16>>, Vec<Vec<RGBA16>>) {
        let w = 256_usize;
        let h = 196_usize;

        let mut res: Vec<Vec<RGBA16>> = Vec::new();
        let mut orig: Vec<Vec<RGBA16>> = Vec::new();

        (0 .. h).for_each(|y| {
            let mut line: Vec<RGBA16> = Vec::new();
            let mut oline: Vec<RGBA16> = Vec::new();

            (0 .. w).for_each(|x| {
                line.push((
                        ((((x * y) as u8) as u16) << 8) + 1,
                        (((((255 - x) * y) as u8) as u16) << 8) + 2,
                        ((((x * (255 - y)) as u8) as u16) << 8) + 3,
                        (((((255 - x) * (255 - y)) as u8) as u16) << 8) + 4
                ));
                oline.push((
                        ((((x * y) as u8) as u16) << 8) + 1,
                        (((((255 - x) * y) as u8) as u16) << 8) + 2,
                        ((((x * (255 - y)) as u8) as u16) << 8) + 3,
                        (((((255 - x) * (255 - y)) as u8) as u16) << 8) + 4
                ));
            });

            res.push(line);
            orig.push(oline);
        });

        (orig, res)
    }

    fn image_rgb() -> (Vec<Vec<RGBA16>>, Vec<Vec<RGB>>) {
        let w = 256_usize;
        let h = 196_usize;

        let mut res_orig: Vec<Vec<RGBA16>> = Vec::new();
        let mut res_data: Vec<Vec<RGB>> = Vec::new();

        (0 .. h).for_each(|y| {
            let mut line_orig: Vec<RGBA16> = Vec::new();
            let mut line_data: Vec<RGB> = Vec::new();

            (0 .. w).for_each(|x| {
                let pix: RGB = (
                    (x * y) as u8,
                    ((255 - x) * y) as u8,
                    (x * (255 - y)) as u8,
                );

                line_orig.push((
                    pix.0 as u16 | (pix.0 as u16) << 8,
                    pix.1 as u16 | (pix.1 as u16) << 8,
                    pix.2 as u16 | (pix.2 as u16) << 8,
                    0xffff_u16
                ));

                line_data.push(pix);
            });

            res_orig.push(line_orig);
            res_data.push(line_data);
        });

        (res_orig, res_data)
    }

    fn image_rgb_16() -> (Vec<Vec<RGBA16>>, Vec<Vec<RGB16>>) {
        let w = 256_usize;
        let h = 196_usize;

        let mut res_orig: Vec<Vec<RGBA16>> = Vec::new();
        let mut res_data: Vec<Vec<RGB16>> = Vec::new();

        (0 .. h).for_each(|y| {
            let mut line_orig: Vec<RGBA16> = Vec::new();
            let mut line_data: Vec<RGB16> = Vec::new();

            (0 .. w).for_each(|x| {
                line_orig.push((
                        1 + ((((x * y) as u8) as u16) << 8),
                        2 + ((((((255 - x) * y) as u8) as u16)) << 8),
                        3 + (((((x * (255 - y)) as u8) as u16)) << 8),
                        0xffff_u16
                ));
                line_data.push((
                        1 + ((((x * y) as u8) as u16) << 8),
                        2 + ((((((255 - x) * y) as u8) as u16)) << 8),
                        3 + (((((x * (255 - y)) as u8) as u16)) << 8)
                ));
            });

            res_orig.push(line_orig);
            res_data.push(line_data);
        });

        (res_orig, res_data)
    }

    fn image_ndx() -> (Vec<Vec<RGBA16>>, // restored image
                       Vec<Vec<u8>>, // @ 0: pal ndx for writer,
                       Vec<RGB>) { // palette
        let w = 256_usize;
        let h = 196_usize;

        let mut pal = vec![(0, 0, 0); 256];

        let mut orig_img: Vec<Vec<RGBA16>> = Vec::new();
        let mut data_img: Vec<Vec<u8>> = Vec::new();

        (0 .. pal.len()).for_each(|k| {
            pal[k].0 = k as u8;
            pal[k].1 = (64 + k) as u8;
            pal[k].2 = (128 + k) as u8;
        });

        (0 .. h).for_each(|y| {
            let mut orig_line: Vec<RGBA16> = Vec::new();
            let mut data_line: Vec<u8> = Vec::new();

            (0 .. w).for_each(|x| {
                let pndx = (x * y) as u8 as usize;
                data_line.push(pndx as u8);
                orig_line.push((
                    (pal[pndx].0 as u16) << 8,
                    (pal[pndx].1 as u16) << 8,
                    (pal[pndx].2 as u16) << 8,
                    0xffff_u16
                ));
            });
            orig_img.push(orig_line);
            data_img.push(data_line);
        });

        (
            orig_img,
            data_img,
            pal
        )
    }

    fn image_ndxa() -> (Vec<Vec<RGBA16>>, // restored image
                       Vec<Vec<u8>>, // @ 0: pal ndx for writer,
                       Vec<RGBA>) { // palette
        let w = 256_usize;
        let h = 196_usize;

        let mut pal = vec![(0, 0, 0, 0); 256];

        let mut orig_img: Vec<Vec<RGBA16>> = Vec::new();
        let mut data_img: Vec<Vec<u8>> = Vec::new();

        (0 .. pal.len()).for_each(|k| {
            pal[k].0 = k as u8;
            pal[k].1 = (64 + k) as u8;
            pal[k].2 = (128 + k) as u8;
            pal[k].3 = (128_i16 - k as i16) as u8;
        });

        (0 .. h).for_each(|y| {
            let mut orig_line: Vec<RGBA16> = Vec::new();
            let mut data_line: Vec<u8> = Vec::new();

            (0 .. w).for_each(|x| {
                let pndx = (x * y) as u8 as usize;
                data_line.push(pndx as u8);
                orig_line.push((
                    (pal[pndx].0 as u16) << 8,
                    (pal[pndx].1 as u16) << 8,
                    (pal[pndx].2 as u16) << 8,
                    (pal[pndx].3 as u16) << 8
                ));
            });
            orig_img.push(orig_line);
            data_img.push(data_line);
        });

        (
            orig_img,
            data_img,
            pal
        )
    }

    //fn progress(_cur: usize, _max: usize, _descr: &String) {
    //}

    //#[test]
    //pub fn test_rgba_none() {
        //let image = image_rgba();

        //write_apng(&"tmp/rgba_None.png".to_string(),
            //&ImageData::RGBA(vec![image]),
            //Some(PkEst::None),
            //progress
        //).unwrap();

        //let back = read_png(&"tmp/rgba_None.png".to_string()).unwrap();

        //assert_eq!(back.width, 256);
        //assert_eq!(back.height, 196);
        //assert_eq!(back.color_type, ColorType::RGBA);
        //assert_eq!(back.data, image);
    //}

    #[test]
    pub fn test_rgba_all() {
        let (orig, image) = image_rgba();

        let types = vec![
            PkEst::None,
            PkEst::Sub,
            PkEst::Up,
            PkEst::Avg,
            PkEst::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgba_{est:?}.png");

            write_apng(&fname,
                &ImageData::RGBA(vec![image.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            let back = read_png(&fname).unwrap();

            assert_eq!(back.width, 256);
            assert_eq!(back.height, 196);
            assert_eq!(back.color_type, ColorType::RGBA);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGBA(vec![image.clone()]));
        });
    }

    #[test]
    pub fn test_rgba_16_all() {
        let (orig, image) = image_rgba_16();

        let types = vec![
            PkEst::None,
            PkEst::Sub,
            PkEst::Up,
            PkEst::Avg,
            PkEst::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgba_16_{est:?}.png");
            let fname_a7 = format!("tmp/rgba_16_{est:?}_a7.png");

            write_apng(&fname,
                &ImageData::RGBA16(vec![image.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            write_apng(&fname_a7,
                &ImageData::RGBA16(vec![image.clone()]),
                Some(*est),
                None,
                true
            ).unwrap();

            let back = read_png(&fname).unwrap();

            assert_eq!(back.width, 256);
            assert_eq!(back.height, 196);
            assert_eq!(back.color_type, ColorType::RGBA16);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGBA16(vec![image.clone()]));
        });
    }

    #[test]
    pub fn test_rgb_all() {
        let (orig, data) = image_rgb();

        let types = vec![
            PkEst::None,
            PkEst::Sub,
            PkEst::Up,
            PkEst::Avg,
            PkEst::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgb_{est:?}.png");

            write_apng(&fname,
                &ImageData::RGB(vec![data.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            let back = read_png(&fname).unwrap();

            assert_eq!(back.width, 256);
            assert_eq!(back.height, 196);
            assert_eq!(back.color_type, ColorType::RGB);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGB(vec![data.clone()]));
        });
    }

    #[test]
    pub fn test_rgb_16_all() {
        let (orig, data) = image_rgb_16();

        let types = vec![
            PkEst::None,
            PkEst::Sub,
            PkEst::Up,
            PkEst::Avg,
            PkEst::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgb_16_{est:?}.png");

            write_apng(&fname,
                &ImageData::RGB16(vec![data.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            let back = read_png(&fname).unwrap();

            assert_eq!(back.width, 256);
            assert_eq!(back.height, 196);
            assert_eq!(back.color_type, ColorType::RGB16);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGB16(vec![data.clone()]));
        });
    }

    #[test]
    pub fn test_ndx_all() {
        let (orig, data, pal) = image_ndx();

        let types = vec![
            PkEst::None,
            PkEst::Sub,
            PkEst::Up,
            PkEst::Avg,
            PkEst::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/ndx_{est:?}.png");

            write_apng(&fname,
                &ImageData::NDX(vec![data.clone()], pal.clone()),
                Some(*est),
                None,
                false
            ).unwrap();

            let back = read_png(&fname).unwrap();

            assert_eq!(back.width, 256);
            assert_eq!(back.height, 196);
            assert_eq!(back.color_type, ColorType::NDX);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::NDX(vec![data.clone()], pal.clone()));
        });
    }

    #[test]
    pub fn test_ndxa_all() {
        let (orig, data, pal) = image_ndxa();

        let types = vec![
            PkEst::None,
            PkEst::Sub,
            PkEst::Up,
            PkEst::Avg,
            PkEst::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/ndxa_{est:?}.png");

            write_apng(&fname,
                &ImageData::NDXA(vec![data.clone()], pal.clone()),
                Some(*est),
                None,
                false
            ).unwrap();

            let back = read_png(&fname).unwrap();

            assert_eq!(back.width, 256);
            assert_eq!(back.height, 196);
            assert_eq!(back.color_type, ColorType::NDXA);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::NDXA(vec![data.clone()], pal.clone()));
        });
    }
}
