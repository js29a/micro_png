//! micro_png
//!
//! Examples: <https://github.com/js29a/micro_png/wiki>

use std::fs::File;
use std::io::{Write, Read};
use std::iter::zip;
use std::fs;

use std::collections::HashMap;

use async_std::task::spawn;
use futures::future::join_all;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use flate2::read::ZlibDecoder;
use crc32fast::Hasher;
use quicklz::{compress, CompressionLevel};

/// RGB 8 bits.
pub type RGB = (u8, u8, u8);

/// RGB + Alpha 8 bits.
pub type RGBA = (u8, u8, u8, u8);

/// RGB 16 bits.
pub type RGB16 = (u16, u16, u16);

/// RGB + Alpha 16 bits.
pub type RGBA16 = (u16, u16, u16, u16);

/// Palette index.
pub type NDX = u8;

/// Palette type (see [ImageData::NDX] and [ImageData::NDXA]).
#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum Palette {
/// One-bit palette - 2 colors.
    P1,
/// Two-bit palette - 4 colors.
    P2,
/// Four-bit palette - 16 colors.
    P4,
/// Eight-bit palette - 256 colors.
    P8
}

/// Grayscale bits count (see [ImageData::GRAY] and [ImageData::GRAYA]).
#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum Grayscale {
/// One-bit - 2 levels, without alpha.
    G1,
/// Two-bit - 4 levels, without alpha.
    G2,
/// Four-bit - 16 levels, without alpha.
    G4,
/// Eigth-bit - 256 levels, with or without alpha.
    G8,
/// Sixteen-bit - 65536 levels, with or without alpha.
    G16
}

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

/// Filter mode.
#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum Filter {
/// No filter.
    None,
/// Pixel on left filter.
    Sub,
/// Pixel above filter.
    Up,
/// Average of pixel above and pixel on left.
    Avg,
/// Paeth filter.
    Paeth
}

/// Color type.
#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum ColorType {
/// RGB 8 bits - [ImageData::RGB].
    RGB,
/// RGB + Alpha 8 bits - [ImageData::RGBA].
    RGBA,
/// Indexed mode - 2, 4, 16 or 256 colors - [ImageData::NDX].
    NDX(Palette),
/// Indexed mode + Alpha - 2, 4, 16 or 256 colors - [ImageData::NDXA].
    NDXA(Palette),
/// RGB 16 bits - [ImageData::RGB16].
    RGB16,
/// RGB + Alpha 16 bits - [ImageData::RGBA16].
    RGBA16,
/// Grayscale without Alpha - [ImageData::GRAY].
    GRAY(Grayscale),
/// Grayscale with Alpha - [ImageData::GRAYA].
///
/// # Note
///
/// The specification covers only 8 and 16 bits here.
    GRAYA(Grayscale)
}

/// Image data. Input for [write_apng]. For loaded image can be accessed using [Image::raw].
///
/// # Data layout
///
/// In descending order: frames then frame rows then row pixels.
///
/// See [write_apng] and [read_png] for details.
#[derive(Eq, Hash, PartialEq, Debug, Clone)]
pub enum ImageData {
/// 24 bit color mode.
    RGB(Vec<Vec<Vec<RGB>>>),
/// 32 bit color mode.
    RGBA(Vec<Vec<Vec<RGBA>>>),
/// 2,4,16,256-color palette without Alpha.
    NDX(Vec<Vec<Vec<NDX>>>, Vec<RGB>, Palette),
/// 2,4,16,256-color palette with Alpha for each palette entry.
    NDXA(Vec<Vec<Vec<NDX>>>, Vec<RGBA>, Palette),
/// 48 bit color mode.
    RGB16(Vec<Vec<Vec<RGB16>>>),
/// 64 bit color mode.
    RGBA16(Vec<Vec<Vec<RGBA16>>>),
/// Grayscale without Alpha.
    GRAY(Vec<Vec<Vec<u16>>>, Grayscale),
/// Grayscale with Alpha.
///
/// # Note
///
/// The specification covers only 8 and 16 bits here.
    GRAYA(Vec<Vec<Vec<(u16, u16)>>>, Grayscale),
}

/// Image structure.
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
/// Color type getter.
    pub fn color_type(&self) -> ColorType {
        self.color_type
    }

/// Image width getter.
    pub fn width(&self) -> usize {
        self.width
    }

/// Image height getter.
    pub fn height(&self) -> usize {
        self.height
    }

/// Image data getter - always RGB + Alpha 16 bits.
    pub fn data(&self) -> Vec<Vec<RGBA16>> {
        self.data.clone()
    }

/// Raw image data getter.
    pub fn raw(&self) -> &ImageData {
        &self.raw
    }

/// Image metadata getter.
    pub fn meta(&self) -> &HashMap<String, String> {
        &self.meta
    }
}

/// APNG builder structure. For explanations see [build_apng].
///
/// # Notes
///
/// Each change takes ownership. This is to avoid unneeded memory clones.
///
/// # Example
///
/// ```rust
///    use micro_png::{APNGBuilder, build_apng, ImageData, RGBA};
///
///    let data: Vec<Vec<RGBA>> = vec![
///        vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 1st line
///        vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 2nd line
///    ];
///
///    let builder = APNGBuilder::new("tmp/foo.png", ImageData::RGBA(vec![data]))
///        .set_adam_7(true);
///
///    build_apng(builder).unwrap();
///
///    // builder variable is not accessible here
/// ```
#[derive(Debug, Clone)]
pub struct APNGBuilder {
    fname: String,
    image_data: ImageData,
    filter: Option<Filter>,
    progress: Option<APNGProgress>,
    adam_7: bool,
    repeat: u32,
    def_dur: (u16, u16),
    dur: HashMap<u32, (u16, u16)>,
    meta: HashMap<String, String>,
    zmeta: HashMap<String, String>
}

impl APNGBuilder {
/// Simple builder ctor.
    pub fn new(fname: &str, image_data: ImageData) -> Self {
        let mut meta: HashMap<String, String> = HashMap::new();

        meta.insert("Comment".to_string(), "created by nobody / SQ6KBQ".to_string());

        Self {
            fname: fname.to_string(),
            image_data,
            filter: None,
            progress: None,
            adam_7: false,
            repeat: 0,
            def_dur: (1, 25),
            dur: HashMap::new(),
            meta,
            zmeta: HashMap::new()
        }
    }

/// Force using a filter.
    pub fn set_filter(mut self, new_filter: Filter) -> Self {
        self.filter = Some(new_filter);
        self
    }

/// Clear using filter filter.
    pub fn clear_filter(mut self) -> Self {
        self.filter = None;
        self
    }

/// Set progress callback.
    pub fn set_progress(mut self, new_progress: APNGProgress) -> Self {
        self.progress = Some(new_progress);
        self
    }

/// Clear progress callback.
    pub fn clear_progress(mut self) -> Self {
        self.progress = None;
        self
    }

/// Set Adam7 flag.
    pub fn set_adam_7(mut self, adam_7: bool) -> Self {
        self.adam_7 = adam_7;
        self
    }

/// Set repeat count.
    pub fn set_repeat(mut self, repeat: u32) -> Self {
        self.repeat = repeat;
        self
    }

/// Set default fame duration.
    pub fn set_def_dur(mut self, def_dur: (u16, u16)) -> Self {
        self.def_dur = def_dur;
        self
    }

/// Set per frame duration.
    pub fn set_dur(mut self, frame: u32, dur: (u16, u16)) -> Self {
        self.dur.insert(frame, dur);
        self
    }

/// Set plaintext metadata.
    pub fn set_meta(mut self, key: &str, value: &str) -> Self {
        self.meta.insert(key.to_string(), value.to_string());
        self
    }

/// Set zlibed metadata.
    pub fn set_zmeta(mut self, key: &str, value: &str) -> Self {
        self.zmeta.insert(key.to_string(), value.to_string());
        self
    }
}

/// Write progress callback.
pub type APNGProgress = fn (cur: usize, total: usize, descr: &str);

type Pred = fn (line: &[RGBA16], above: &[RGBA16], color_type: ColorType) -> Vec<u8>;

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

fn pack_pix(color_type: ColorType, row: &[RGBA16]) -> Vec<RGBA16> {
    match color_type {
        ColorType::NDXA(Palette::P8) | ColorType::NDX(Palette::P8) =>
            row.to_vec(),
        ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) => {
            assert!((row.len() & 1) == 0);
            let mut res: Vec<RGBA16> = Vec::new();
            (0 .. row.len()).step_by(2).for_each(|ndx| {
                res.push((
                      ((row[ndx    ].0 & 15) << 4)
                    | ( row[ndx + 1].0 & 15      ),
                    0,
                    0,
                    0
                ));
            });
            res
        },
        ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) => {
            assert!((row.len() & 3) == 0);
            let mut res: Vec<RGBA16> = Vec::new();
            (0 .. row.len()).step_by(4).for_each(|ndx| {
                res.push((
                      ((row[ndx    ].0 & 3) << 6)
                    | ((row[ndx + 1].0 & 3) << 4)
                    | ((row[ndx + 2].0 & 3) << 2)
                    | ( row[ndx + 3].0 & 3      ),
                    0,
                    0,
                    0
                ));
            });
            res
        },
        ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) => {
            assert!((row.len() & 7) == 0);
            let mut res: Vec<RGBA16> = Vec::new();
            (0 .. row.len()).step_by(8).for_each(|ndx| {
                res.push((
                      ((row[ndx    ].0 & 1) << 7)
                    | ((row[ndx + 1].0 & 1) << 6)
                    | ((row[ndx + 2].0 & 1) << 5)
                    | ((row[ndx + 3].0 & 1) << 4)
                    | ((row[ndx + 4].0 & 1) << 3)
                    | ((row[ndx + 5].0 & 1) << 2)
                    | ((row[ndx + 6].0 & 1) << 1)
                    | ( row[ndx + 7].0 & 1      ),
                    0,
                    0,
                    0
                ));
            });
            res
        },
        ColorType::GRAY(Grayscale::G4) => {
            assert!((row.len() & 1) == 0);
            let mut res: Vec<RGBA16> = Vec::new();
            (0 .. row.len()).step_by(2).for_each(|ndx| {
                res.push((
                      ((row[ndx    ].0 & 15) << 4)
                    | ( row[ndx + 1].0 & 15      ),
                    0,
                    0,
                    0
                ));
            });
            res
        },
        ColorType::GRAY(Grayscale::G2) => {
            assert!((row.len() & 3) == 0);
            let mut res: Vec<RGBA16> = Vec::new();
            (0 .. row.len()).step_by(4).for_each(|ndx| {
                res.push((
                      ((row[ndx    ].0 & 3) << 6)
                    | ((row[ndx + 1].0 & 3) << 4)
                    | ((row[ndx + 2].0 & 3) << 2)
                    | ( row[ndx + 3].0 & 3      ),
                    0,
                    0,
                    0
                ));
            });
            res
        },
        ColorType::GRAY(Grayscale::G1) => {
            assert!((row.len() & 7) == 0);
            let mut res: Vec<RGBA16> = Vec::new();
            (0 .. row.len()).step_by(8).for_each(|ndx| {
                res.push((
                      ((row[ndx    ].0 & 1) << 7)
                    | ((row[ndx + 1].0 & 1) << 6)
                    | ((row[ndx + 2].0 & 1) << 5)
                    | ((row[ndx + 3].0 & 1) << 4)
                    | ((row[ndx + 4].0 & 1) << 3)
                    | ((row[ndx + 5].0 & 1) << 2)
                    | ((row[ndx + 6].0 & 1) << 1)
                    | ( row[ndx + 7].0 & 1      ),
                    0,
                    0,
                    0
                ));
            });
            res
        }
        a => panic!("bad pack_pix argument: {a:?}"),
    }
}

fn png_none(row_raw: &[RGBA16], _above: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();

    res.push(0_u8);

    let row = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, row_raw),
        _ => row_raw.to_vec(),
    };

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
            ColorType::NDXA(_) | ColorType::NDX(_) => vec![
                pix.0 as u8
            ],
            ColorType::GRAY(Grayscale::G8) => vec![
                pix.0 as u8
            ],
            ColorType::GRAYA(Grayscale::G8) => vec![
                pix.0 as u8,
                pix.3 as u8
            ],
            ColorType::GRAY(Grayscale::G16) => vec![
                (pix.0 >> 8) as u8,
                (pix.0 & 0xff) as u8,
            ],
            ColorType::GRAYA(Grayscale::G16) => vec![
                (pix.0 >> 8) as u8,
                (pix.0 & 0xff) as u8,
                (pix.3 >> 8) as u8,
                (pix.3 & 0xff) as u8,
            ],
            ColorType::GRAYA(_) | ColorType::GRAY(_) => vec![// NOTE already packed
                pix.0 as u8
            ]
        }
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_sub(row_raw: &[RGBA16], _above: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();
    let mut prev: RGBA16 = (0, 0, 0, 0);

    res.push(1_u8);

    let row = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, row_raw),
        _ => row_raw.to_vec(),
    };

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
            ColorType::NDXA(_) | ColorType::NDX(_) => vec![
                sub((pix.0 & 0xff) as u8, (prev.0 & 0xff) as u8),
            ],
            ColorType::GRAY(Grayscale::G8) => vec![
                sub(pix.0 as u8, prev.0 as u8),
            ],
            ColorType::GRAYA(Grayscale::G8) => vec![
                sub(pix.0 as u8, prev.0 as u8),
                sub(pix.3 as u8, prev.3 as u8),
            ],
            ColorType::GRAY(Grayscale::G16) => vec![
                sub((pix.0 >> 8) as u8, (prev.0 >> 8) as u8),
                sub((pix.0 & 0xff) as u8, (prev.0 & 0xff) as u8),
            ],
            ColorType::GRAYA(Grayscale::G16) => vec![
                sub((pix.0 >> 8) as u8, (prev.0 >> 8) as u8),
                sub((pix.0 & 0xff) as u8, (prev.0 & 0xff) as u8),
                sub((pix.3 >> 8) as u8, (prev.3 >> 8) as u8),
                sub((pix.3 & 0xff) as u8, (prev.3 & 0xff) as u8),
            ],
            ColorType::GRAYA(_) | ColorType::GRAY(_) => vec![// NOTE already packed
                sub((pix.0 & 0xff) as u8, (prev.0 & 0xff) as u8),
            ],
        };
        prev = *pix;
        r
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_up(row_raw: &[RGBA16], above_raw: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();

    res.push(2_u8);

    let row = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, row_raw),
        _ => row_raw.to_vec(),
    };

    let above = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, above_raw),
        _ => above_raw.to_vec(),
    };

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
            ColorType::NDXA(_) | ColorType::NDX(_) =>
                vec![
                    sub(pix.0 as u8, above[x].0 as u8)
                ],
            ColorType::GRAYA(Grayscale::G8) =>
                vec![
                    sub(pix.0 as u8, above[x].0 as u8),
                    sub(pix.3 as u8, above[x].3 as u8),
                ],
            ColorType::GRAY(Grayscale::G8) =>
                vec![
                    sub(pix.0 as u8, above[x].0 as u8),
                ],
            ColorType::GRAYA(Grayscale::G16) =>
                vec![
                    sub((pix.0 >> 8) as u8, (above[x].0 >> 8) as u8),
                    sub((pix.0 & 0xff) as u8, (above[x].0 & 0xff) as u8),
                    sub((pix.3 >> 8) as u8, (above[x].3 >> 8) as u8),
                    sub((pix.3 & 0xff) as u8, (above[x].3 & 0xff) as u8),
                ],
            ColorType::GRAY(Grayscale::G16) =>
                vec![
                    sub((pix.0 >> 8) as u8, (above[x].0 >> 8) as u8),
                    sub((pix.0 & 0xff) as u8, (above[x].0 & 0xff) as u8),
                ],
            ColorType::GRAYA(_) | ColorType::GRAY(_) =>
                vec![ // NOTE already packed
                    sub(pix.0 as u8, above[x].0 as u8),
                ],
        }
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_avg(row_raw: &[RGBA16], above_raw: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    assert_eq!(row_raw.len(), above_raw.len());

    let mut res: Vec<u8> = Vec::new();
    let mut prev: RGBA16 = (0, 0, 0, 0);

    res.push(3_u8);

    let row = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, row_raw),
        _ => row_raw.to_vec(),
    };

    let above = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, above_raw),
        _ => above_raw.to_vec(),
    };

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
            ColorType::NDXA(_) | ColorType::NDX(_) => vec![
                sub(pix.0 as u8, a0l),
            ],
            ColorType::GRAY(Grayscale::G8) => vec![
                sub(pix.0 as u8, a0l),
            ],
            ColorType::GRAYA(Grayscale::G8) => vec![
                sub(pix.0 as u8, a0l),
                sub(pix.3 as u8, a3l),
            ],
            ColorType::GRAY(Grayscale::G16) => vec![
                sub((pix.0 >> 8) as u8, a0h),
                sub((pix.0 & 0xff) as u8, a0l),
            ],
            ColorType::GRAYA(Grayscale::G16) => vec![
                sub((pix.0 >> 8) as u8, a0h),
                sub((pix.0 & 0xff) as u8, a0l),
                sub((pix.3 >> 8) as u8, a3h),
                sub((pix.3 & 0xff) as u8, a3l),
            ],
            ColorType::GRAYA(_) | ColorType::GRAY(_) => vec![ // NOTE already packed
                sub(pix.0 as u8, a0l),
            ],
        };

        prev = *pix;
        r
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_paeth(row_raw: &[RGBA16], above_raw: &[RGBA16], color_type: ColorType) -> Vec<u8> {
    assert_eq!(row_raw.len(), above_raw.len());

    let mut res: Vec<u8> = Vec::new();
    let mut prev: RGBA16 = (0, 0, 0, 0);

    res.push(4_u8);

    let row = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, row_raw),
        _ => row_raw.to_vec(),
    };

    let above = match color_type {
        ColorType::GRAY(Grayscale::G4) |
        ColorType::GRAY(Grayscale::G2) |
        ColorType::GRAY(Grayscale::G1) |
        ColorType::NDXA(_) | ColorType::NDX(_) =>
            pack_pix(color_type, above_raw),
        _ => above_raw.to_vec(),
    };

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
            ColorType::NDXA(_) | ColorType::NDX(_) => vec![
                sub((pix.0 & 0xff) as u8, paeth(a0l, b0l, c0l)),
            ],
            ColorType::GRAY(Grayscale::G8) => vec![
                sub(pix.0 as u8, paeth(a0l, b0l, c0l)),
            ],
            ColorType::GRAYA(Grayscale::G8) => vec![
                sub(pix.0 as u8, paeth(a0l, b0l, c0l)),
                sub(pix.3 as u8, paeth(a3l, b3l, c3l)),
            ],
            ColorType::GRAY(Grayscale::G16) => vec![
                sub((pix.0 >> 8) as u8, paeth(a0h, b0h, c0h)),
                sub((pix.0 & 0xff) as u8, paeth(a0l, b0l, c0l)),
            ],
            ColorType::GRAYA(Grayscale::G16) => vec![
                sub((pix.0 >> 8) as u8, paeth(a0h, b0h, c0h)),
                sub((pix.0 & 0xff) as u8, paeth(a0l, b0l, c0l)),
                sub((pix.3 >> 8) as u8, paeth(a3h, b3h, c3h)),
                sub((pix.3 & 0xff) as u8, paeth(a3l, b3l, c3l)),
            ],
            ColorType::GRAYA(_) | ColorType::GRAY(_) => vec![ // NOTE already packed
                sub((pix.0 & 0xff) as u8, paeth(a0l, b0l, c0l)),
            ],
        };

        prev = *pix;
        r
    }).collect::<Vec<Vec<u8>>>().iter().flatten());

    res
}

fn png_rev(left: u8, up: u8, corner: u8, est: Filter) -> u8 {
    match est {
        Filter::None => 0,
        Filter::Sub => left,
        Filter::Up => up,
        Filter::Avg => ((left as u16 + up as u16) >> 1) as u8,
        Filter::Paeth => paeth(left, up, corner)
    }
}

fn pk_size(payload: &[u8]) -> usize {
    compress(payload, CompressionLevel::Lvl1).len() // NOTE quick estimation
}

async fn estimate_worker(est: Filter, so_far: Vec<u8>, payload: Vec<u8>) -> (Filter, usize, Vec<u8>) {
    async fn doit(_so_far: Vec<u8>, payload: Vec<u8>) -> usize {
        pk_size(&payload[..])
    }

    (est, spawn(doit(so_far, payload.clone())).await, payload)
}

fn elect_best(preds: &[(Filter, Pred)], so_far: Vec<u8>, line: &[RGBA16], above: &[RGBA16], color_type: ColorType) ->
    (Filter, Vec<u8>) {

    let tasks = preds.iter().map(|(id, func)| {
        estimate_worker(*id, so_far.clone(), func(line, above, color_type))
    });

    let output = futures_executor::block_on(join_all(tasks));

    let mut best = Filter::None;
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
        ImageData::NDXA(ndx, _, _) | ImageData::NDX(ndx, _, _) =>
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
            }).collect(),
        ImageData::GRAYA(gray, _) =>
            gray.iter().map(|frame| -> Vec<Vec<RGBA16>> {
                frame.iter().map(|line| -> Vec<RGBA16> {
                    line.iter().map(|pix| -> RGBA16 {
                        (
                            pix.0,
                            0,
                            0,
                            pix.1
                        )
                    }).collect()
                }).collect()
            }).collect(),
        ImageData::GRAY(gray, _) =>
            gray.iter().map(|frame| -> Vec<Vec<RGBA16>> {
                frame.iter().map(|line| -> Vec<RGBA16> {
                    line.iter().map(|pix| -> RGBA16 {
                        (
                            *pix,
                            0,
                            0,
                            0
                        )
                    }).collect()
                }).collect()
            }).collect(),
    }
}

fn gen_palette(image_data: &ImageData) -> Vec<u8> {
    let mut res: Vec<u8> = Vec::new();

    let pal_len = match image_data {
        ImageData::NDXA(_, _, Palette::P8) | ImageData::NDX(_, _, Palette::P8) => 256,
        ImageData::NDXA(_, _, Palette::P4) | ImageData::NDX(_, _, Palette::P4) => 16,
        ImageData::NDXA(_, _, Palette::P2) | ImageData::NDX(_, _, Palette::P2) => 4,
        ImageData::NDXA(_, _, Palette::P1) | ImageData::NDX(_, _, Palette::P1) => 2,
        _ => 0
    };

    match image_data {
        ImageData::NDXA(_, pal, _) => {
            let mut plte: Vec<u8> = b"PLTE".to_vec();

            plte.extend(pal[0 .. pal_len].iter().map(|rgba: &RGBA| {
                vec![
                    rgba.0,
                    rgba.1,
                    rgba.2
                ]
            }).collect::<Vec<Vec<u8>>>().iter().flatten());

            res.extend(png_chunk(&plte));

            let mut trns: Vec<u8> = b"tRNS".to_vec();

            trns.extend(pal[0 .. pal_len].iter().map(|rgba: &RGBA| {
                rgba.3
            }).collect::<Vec<u8>>());

            res.extend(png_chunk(&trns));
        },
        ImageData::NDX(_, pal, _) => {
            let mut plte: Vec<u8> = b"PLTE".to_vec();

            plte.extend(pal[0 .. pal_len].iter().map(|rgb: &RGB| {
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
    stats: &mut Stats, ndx: u32, filter: Option<Filter>, seq: &mut u32, frames: &Vec<Vec<Vec<RGBA16>>>,
    adam_7: bool) -> Vec<u8> {

    let preds_first: Vec<(Filter, Pred)> = vec![
        (Filter::None, png_none),
        (Filter::Sub, png_sub)
    ];

    let preds_next: Vec<(Filter, Pred)> = vec![
        (Filter::None, png_none),
        (Filter::Sub, png_sub),
        (Filter::Up, png_up),
        (Filter::Avg, png_avg),
        (Filter::Paeth, png_paeth)
    ];

    let mut payload: Vec<u8> = Vec::new();
    let mut above: Vec<RGBA16> = vec![];

    let mut fdat: Vec<u8> = b"fdAT".to_vec();

    if ndx == 0 {
        fdat = b"IDAT".to_vec();
        *seq += 1;
    }
    else {
        *seq += 1;
        fdat.extend((*seq - 2).to_be_bytes());
    }

    let frame = &frames[ndx as usize];
    let mut first = true;

    let a_lo = if adam_7 { 1 } else { 0 };
    let a_hi = if adam_7 { 7 } else { 0 };

    (a_lo ..= a_hi).for_each(|a| {
        first = true;
        above = vec![];

        if adam_7 {
            if let Some(p) = progress {
                p(a - 1, a_hi - a_lo + 1, "Adam7");
            }
        }

        zip(0 .. frame.len(), frame.iter()).for_each(|(y, row_raw)| {
            if !adam_7 {
                if let Some(p) = progress {
                    p(y, frame.len(), "lines");
                }
            }

            let mut so_far: Vec<u8> = vec![];

            let mut line = if adam_7 && frames.len() == 1 {
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

            match color_type {
                ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) => {
                    while (line.len() & 7) != 0 {
                        line.push((0, 0, 0, 0))
                    }
                },
                ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) => {
                    while (line.len() & 3) != 0 {
                        line.push((0, 0, 0, 0))
                    }
                },
                ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) => {
                    while (line.len() & 1) != 0 {
                        line.push((0, 0, 0, 0))
                    }
                },
                ColorType::GRAY(Grayscale::G1) => {
                    while (line.len() & 7) != 0 {
                        line.push((0, 0, 0, 0))
                    }
                },
                ColorType::GRAY(Grayscale::G2) => {
                    while (line.len() & 3) != 0 {
                        line.push((0, 0, 0, 0))
                    }
                },
                ColorType::GRAY(Grayscale::G4) => {
                    while (line.len() & 1) != 0 {
                        line.push((0, 0, 0, 0))
                    }
                },
                _ => ()
            };

            if first {
                match filter {
                    Some(Filter::None) => {
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
                            Filter::None => stats.n_none += 1,
                            Filter::Sub => stats.n_sub += 1,
                            _ => panic!("pred elect @ one line: failure")
                        }
                    }
                }
                first = false;
            }
            else {
                match filter {
                    Some(Filter::None) => {
                        stats.n_none += 1;
                        payload.extend(png_none(&line[..], &above[..], color_type));
                    },
                    Some(Filter::Sub) => {
                        stats.n_sub += 1;
                        payload.extend(png_sub(&line[..], &above[..], color_type));
                    },
                    Some(Filter::Up) => {
                        stats.n_up += 1;
                        payload.extend(png_up(&line[..], &above[..], color_type));
                    },
                    Some(Filter::Avg) => {
                        stats.n_avg += 1;
                        payload.extend(png_avg(&line[..], &above[..], color_type));
                    },
                    Some(Filter::Paeth) => {
                        stats.n_paeth += 1;
                        payload.extend(png_paeth(&line[..], &above[..], color_type));
                    },
                    None => { // NOTE elect
                        let (best, bpayload) = elect_best(&preds_next, so_far.clone(), &line, &above, color_type);

                        so_far.extend(&bpayload);
                        payload.extend(bpayload);

                        match best {
                            Filter::None => stats.n_none += 1,
                            Filter::Sub => stats.n_sub += 1,
                            Filter::Up => stats.n_up += 1,
                            Filter::Avg => stats.n_avg += 1,
                            Filter::Paeth => stats.n_paeth += 1
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

/// Create (A)PNG binary using builder. Output: `Vec<u8>`.
pub fn build_apng_u8(builder: APNGBuilder) -> Result<Vec<u8>, String> {
    let image_data = &builder.image_data;
    let filter = builder.filter;
    let progress = builder.progress;
    let mut adam_7 = builder.adam_7;

    let color_type = match image_data {
        ImageData::RGBA16(_) => ColorType::RGBA16,
        ImageData::RGB16(_) => ColorType::RGB16,
        ImageData::RGBA(_) => ColorType::RGBA,
        ImageData::RGB(_) => ColorType::RGB,
        ImageData::NDXA(_, _, p) => ColorType::NDXA(*p),
        ImageData::NDX(_, _, p) => ColorType::NDX(*p),
        ImageData::GRAYA(_, g) => ColorType::GRAYA(*g),
        ImageData::GRAY(_, g) => ColorType::GRAY(*g),
    };

    match color_type {
        ColorType::GRAYA(Grayscale::G4) |
        ColorType::GRAYA(Grayscale::G2) |
        ColorType::GRAYA(Grayscale::G1) =>
            return Err("forbidden by PNG specification".to_string()),
        _ => ()
    };

    let frames = prepare_frames(image_data);

    if frames.is_empty() {
        return Err("the image is empty".to_string());
    }

    if frames[0].is_empty() {
        return Err("the height is zero".to_string());
    }

    if frames[0][0].is_empty() {
        return Err("the width is zero".to_string());
    }

    let mut err: Option<String> = None;

    let h_ok = frames[0].len();
    let w_ok = frames[0][0].len();

    zip(0 .. frames.len(), frames.iter()).for_each(|(ff, f)| {
        if f.len() != h_ok {
            let l = f.len();
            err = Some(format!("bad frame #{ff} height: {l} instead of {h_ok}"))
        }

        zip(0 .. frames[0].len(), f.iter()).for_each(|(rr, r)| {
            if r.len() != w_ok {
                let l = r.len();
                err = Some(format!("bad line width #{rr} at frame #{ff} height: {l} instead of {w_ok}"))
            }
        });
    });

    if let Some(msg) = err {
        return Err(msg)
    }

    let mut res: Vec<u8> = vec![];

    res.extend(b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a");

    let mut ihdr: Vec<u8> = b"IHDR".to_vec();

    let width = frames[0][0].len();
    let height = frames[0].len();

    ihdr.extend((width as u32).to_be_bytes());
    ihdr.extend((height as u32).to_be_bytes());

    let color_byte = match color_type {
        ColorType::RGBA16 => b'\x06',
        ColorType::RGB16 => b'\x02',
        ColorType::RGBA => b'\x06',
        ColorType::RGB => b'\x02',
        ColorType::NDX(_) => b'\x03',
        ColorType::NDXA(_) => b'\x03',
        ColorType::GRAY(_) => b'\x00',
        ColorType::GRAYA(_) => b'\x04',
    };

    let bpp = match color_type {
        ColorType::RGBA16 => b'\x10',
        ColorType::RGB16 => b'\x10',
        ColorType::RGBA => b'\x08',
        ColorType::RGB => b'\x08',
        ColorType::NDXA(Palette::P8) | ColorType::NDX(Palette::P8) => b'\x08',
        ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) => b'\x04',
        ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) => b'\x02',
        ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) => b'\x01',
        ColorType::GRAY(Grayscale::G1) => b'\x01',
        ColorType::GRAY(Grayscale::G2) => b'\x02',
        ColorType::GRAY(Grayscale::G4) => b'\x04',
        ColorType::GRAY(Grayscale::G8) => b'\x08',
        ColorType::GRAY(Grayscale::G16) => b'\x10',
        ColorType::GRAYA(Grayscale::G8) => b'\x08',
        ColorType::GRAYA(Grayscale::G16) => b'\x10',
        _ => panic!("how???")
    };

    ihdr.append(&mut vec![bpp, color_byte, b'\x00', b'\x00', if adam_7 { b'\x01' } else { b'\x00' }]);

    res.extend(png_chunk(&ihdr));

    res.extend(gen_palette(image_data));

    builder.meta.iter().for_each(|(k, v)| {
        let mut text: Vec<u8> = b"tEXt".to_vec();
        text.extend(k.bytes());
        text.extend(b"\x00");
        text.extend(v.bytes());
        res.extend(png_chunk(&text));
    });

    builder.zmeta.iter().for_each(|(k, v)| {
        let mut text: Vec<u8> = b"zTXt".to_vec();
        text.extend(k.bytes());
        text.extend(b"\x00");
        text.extend(b"\x00");

        let mut e = ZlibEncoder::new(Vec::new(), Compression::best());
        e.write_all(v.as_bytes()).expect("compress");
        let c = e.finish().expect("zlib flush");
        text.extend(c);

        res.extend(png_chunk(&text));
    });

    if frames.len() > 1 {
        let mut actl: Vec<u8> = b"acTL".to_vec();

        actl.extend((frames.len() as u32).to_be_bytes());
        actl.extend(builder.repeat.to_be_bytes());

        res.extend(png_chunk(&actl));
        adam_7 = false;
    }

    let mut seq = 1_u32;

    let mut stats = Stats::default();

    zip(0 .. frames.len() as u32, frames.iter()).for_each(|(ndx, frame)| {
        if frames.len() > 1 && progress.is_some() {
            if let Some(p) = progress {
                p(ndx as usize, frames.len(), "frames");
            }
        }

        if frames.len() > 1 {
            let mut fctl: Vec<u8> = b"fcTL".to_vec();

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

            if let Some(d) = builder.dur.get(&ndx) {
                fctl.extend(d.0.to_be_bytes());
                fctl.extend(d.1.to_be_bytes());
            }
            else {
                fctl.extend(builder.def_dur.0.to_be_bytes());
                fctl.extend(builder.def_dur.1.to_be_bytes());
            }

            fctl.extend((0_u8).to_be_bytes());// dispose, 0 - copy
            fctl.extend((1_u8).to_be_bytes());// blend, 1 - over

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
            filter,
            &mut seq,
            &frames,
            adam_7
        ));
    });

    let iend: Vec<u8> = b"IEND".to_vec();

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

/// Generate APNG bytes. For explanations see [write_apng]. Output: `Vec<u8>`.
pub fn write_apng_u8(image_data: ImageData, filter: Option<Filter>, progress: Option<APNGProgress>, adam_7: bool)
    -> Result<Vec<u8>, String> {

    let mut builder = APNGBuilder::new("", image_data)
        .set_adam_7(adam_7);

    if let Some(f) = filter {
        builder = builder.set_filter(f)
    }

    if let Some(p) = progress {
        builder = builder.set_progress(p)
    }

    build_apng_u8(builder)
}

/// Write (A)PNG file.
///
/// For plain PNG use one frame input.
///
/// # Arguments
///
/// * `fname` - output filename,
/// * `image_data` - the image,
/// * `filter` - force to use a filter,
/// * `progress` - write progress callback,
/// * `adam_7` - flag to use Adam7 output.
///
/// # Example
///
/// ```rust
///    use micro_png::*;
///
///    let data: Vec<Vec<RGBA>> = vec![
///        vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 1st line
///        vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 2nd line
///    ];
///
///    write_apng("tmp/back.png",
///        ImageData::RGBA(vec![data]), // write one frame
///        None ,// automatically select filtering
///        None, // no progress callback
///        false // no Adam-7
///    ).expect("can't save back.png");
/// ```
pub fn write_apng(fname: &str, image_data: ImageData, filter: Option<Filter>,
    progress: Option<APNGProgress>, adam_7: bool) -> Result<(), String> {
    if let Ok(mut f) = File::create(fname) {
        if f.write_all(&write_apng_u8(image_data, filter, progress, adam_7)?).is_err() {
            Err(format!("write error: {fname}"))
        }
        else {
            Ok(())
        }
    }
    else {
        Err(format!("open error: {fname}"))
    }
}

/// Create image file using builder.
///
/// # Example
///
/// ```rust
///    use micro_png::*;
///
///    let data: Vec<Vec<RGBA>> = vec![
///        vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 1st line
///        vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 2nd line
///        vec![(0, 255, 0, 255), (255, 0, 255, 255)],// the 3rd line
///    ];
///
///     build_apng(
///         APNGBuilder::new("tmp/builder-test.png", ImageData::RGBA(vec![data]))
///             .set_meta("Comment", "test comment")
///             .set_zmeta("Author", "test author")
///     ).expect("can't write tmp/builder-test.png");
/// ```
pub fn build_apng(builder: APNGBuilder) -> Result<(), String> {
    let fname = builder.fname.clone();

    if let Ok(mut f) = File::create(&fname) {
        if f.write_all(&build_apng_u8(builder)?).is_err() {
            Err(format!("write error: {fname}"))
        }
        else {
            Ok(())
        }
    }
    else {
        Err(format!("open error: {fname}"))
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

    if 8 + length > inp.len() {
        return Err("chunk underflow".to_string())
    }

    Ok((code, &inp[8 .. 8 + length]))
}

type IdatResult = (Vec<Vec<RGBA16>>, ImageData, Vec<u8>);

fn unpack_idat(width: usize, height: usize, raw: &[u8], color_type: ColorType, pal: &[RGBA])
    -> Result<IdatResult, String> {
    let mut res: Vec<Vec<RGBA16>> = Vec::new();

    let mut raw_rgba16: Vec<Vec<RGBA16>> = Vec::new();
    let mut raw_rgb16: Vec<Vec<RGB16>> = Vec::new();
    let mut raw_rgba: Vec<Vec<RGBA>> = Vec::new();
    let mut raw_rgb: Vec<Vec<RGB>> = Vec::new();
    let mut raw_ndx: Vec<Vec<u8>> = Vec::new();
    let mut raw_gray: Vec<Vec<u16>> = Vec::new();
    let mut raw_graya: Vec<Vec<(u16, u16)>> = Vec::new();

    let slice_len = match color_type {
        ColorType::RGBA16 => width * 4 * 2 + 1,
        ColorType::RGB16 => width * 3 * 2 + 1,
        ColorType::RGBA => width * 4 + 1,
        ColorType::RGB => width * 3 + 1,
        ColorType::NDXA(Palette::P8) | ColorType::NDX(Palette::P8) => width + 1,
        ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) => ((width + 1) / 2) + 1,
        ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) => ((width + 3) / 4) + 1,
        ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) => ((width + 7) / 8) + 1,

        ColorType::GRAY(Grayscale::G1) => (width + 7) / 8 + 1,
        ColorType::GRAY(Grayscale::G2) =>  (width + 3) / 4 + 1,
        ColorType::GRAY(Grayscale::G4) =>  (width + 1) / 2 + 1,

        ColorType::GRAY(Grayscale::G8) => width + 1,
        ColorType::GRAYA(Grayscale::G8) => width * 2 + 1,
        ColorType::GRAY(Grayscale::G16) => width * 2 + 1,
        ColorType::GRAYA(Grayscale::G16) => width * 4 + 1,
        _ => panic!("how???")
    };

    if slice_len * height > raw.len() {
        return Err("unpack_idat underflow".to_string())
    }

    let mut above: Vec<RGBA16> = vec![(0, 0, 0, 0); width];
    let mut ndx_above: Vec<u8> = vec![0; width * 4];// XXX for GRAY(A)
    let mut ndx_line: Vec<u8> = Vec::new();
    let mut ndx_line_raw: Vec<u8> = Vec::new();

    let status = (0 .. height).map(|y| -> Result<(), String> {
        let offs = match color_type {
            ColorType::RGBA16 => (width * 4 * 2 + 1) * y,
            ColorType::RGB16 => (width * 3 * 2 + 1) * y,
            ColorType::RGBA => (width * 4 + 1) * y,
            ColorType::RGB => (width * 3 + 1) * y,
            ColorType::NDXA(Palette::P8) | ColorType::NDX(Palette::P8) => (width + 1) * y,
            ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) => ((width + 1) / 2 + 1) * y,
            ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) => ((width + 3) / 4 + 1) * y,
            ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) => ((width + 7) / 8 + 1) * y,
            ColorType::GRAY(Grayscale::G1) => ((width + 7) / 8 + 1) * y,
            ColorType::GRAY(Grayscale::G2) => ((width + 3) / 4 + 1) * y,
            ColorType::GRAY(Grayscale::G4) => ((width + 1) / 2 + 1) * y,
            ColorType::GRAY(Grayscale::G8) => (width + 1) * y,
            ColorType::GRAYA(Grayscale::G8) => (width * 2 + 1) * y,
            ColorType::GRAY(Grayscale::G16) => (width * 2 + 1) * y,
            ColorType::GRAYA(Grayscale::G16) => (width * 4 + 1) * y,
            _ => panic!("how???")
        };

        if offs + slice_len > raw.len() {
            return Err("unpack_idat: buffer too short".to_string())
        }

        let slice = &raw[offs .. offs + slice_len];
        let mode = slice[0];
        let mut line: Vec<RGBA16> = Vec::new();

        let est = match mode {
            0 => Filter::None,
            1 => Filter::Sub,
            2 => Filter::Up,
            3 => Filter::Avg,
            4 => Filter::Paeth,
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
            ColorType::NDXA(Palette::P8) | ColorType::NDX(Palette::P8) |
            ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) |
            ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) |
            ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) => {
                let line_width = match color_type {
                    ColorType::NDXA(Palette::P8) | ColorType::NDX(Palette::P8) => width,
                    ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) => (width + 1) / 2,
                    ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) => (width + 3) / 4,
                    ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) => (width + 7) / 8,
                    _ => panic!("internal unpack_idat error"),
                };

                ndx_line = Vec::new();
                ndx_line_raw = Vec::new();
                //let mut ndx_line: Vec<u8> = Vec::new();
                let pal_status = (0 .. line_width).map(|ox| -> Result<(), String> {
                    let x = ox;
                    let ndx = add(slice[ox + 1], png_rev(
                            if x < 1 { 0 } else { ndx_line[x - 1]},
                            ndx_above[x],
                            if x < 1 { 0 } else { ndx_above[x - 1]},
                            est
                        )
                    );

                    let (unpacked, top) = match color_type {
                        ColorType::NDXA(Palette::P8) | ColorType::NDX(Palette::P8) =>
                            (vec![
                                ndx
                            ], 1),
                        ColorType::NDXA(Palette::P4) | ColorType::NDX(Palette::P4) =>
                            (vec![
                                (ndx >> 4) & 0xf,
                                ndx & 0xf
                            ], 2),
                        ColorType::NDXA(Palette::P2) | ColorType::NDX(Palette::P2) =>
                            (vec![
                                (ndx >> 6) & 3,
                                (ndx >> 4) & 3,
                                (ndx >> 2) & 3,
                                ndx & 3
                            ], 4),
                        ColorType::NDXA(Palette::P1) | ColorType::NDX(Palette::P1) =>
                            (vec![
                                (ndx >> 7) & 1,
                                (ndx >> 6) & 1,
                                (ndx >> 5) & 1,
                                (ndx >> 4) & 1,
                                (ndx >> 3) & 1,
                                (ndx >> 2) & 1,
                                (ndx >> 1) & 1,
                                ndx & 1,
                            ], 8),
                        _ => panic!("internal error @ unpack_idat")
                    };

                    (0 .. top).for_each(|pos| {
                        if line.len() >= width {
                            return
                        }

                        let ndx1 = unpacked[pos];

                        let pix: RGBA16 = (
                            (pal[ndx1 as usize].0 as u16) << 8,
                            (pal[ndx1 as usize].1 as u16) << 8,
                            (pal[ndx1 as usize].2 as u16) << 8,
                            (pal[ndx1 as usize].3 as u16) << 8
                        );

                        line.push((
                            pix.0,
                            pix.1,
                            pix.2,
                            if let ColorType::NDX(_) = color_type {
                                0xffff_u16
                            }
                            else {
                                pix.3
                            }
                        ));
                        ndx_line_raw.push(ndx1);
                    });
                    ndx_line.push(ndx);
                    Ok(())
                }).find(|x| x.is_err());
                if let Some(e) = pal_status {
                    return e;
                }
                raw_ndx.push(ndx_line_raw.clone());
            },
            ColorType::GRAY(Grayscale::G8) => {
                ndx_line = Vec::new();
                let mut gray_line = Vec::new();

                (0 .. width).step_by(1).for_each(|ox| {
                    let x = ox;
                    let value = add(slice[ox + 1], png_rev(
                        if x < 1 { 0 } else { ndx_line[x - 1] },
                        ndx_above[x],
                        if x < 1 { 0 } else { ndx_above[x - 1]  },
                        est
                    ));

                    ndx_line.push(value);

                    let val = (value as u32 * 0xffff_u32 / 0xff_u32) as u16;

                    gray_line.push(value as u16);

                    line.push((
                        val,
                        val,
                        val,
                        0xffff_u16
                    ));
                });
                raw_gray.push(gray_line);
            },
            ColorType::GRAYA(Grayscale::G8) => {
                ndx_line = Vec::new();
                let mut graya_line = Vec::new();

                (0 .. width * 2).step_by(2).for_each(|ox| {
                    let x = ox;

                    let value = add(slice[ox + 1], png_rev(
                        if x < 2 { 0 } else { ndx_line[x - 2] },
                        ndx_above[x],
                        if x < 2 { 0 } else { ndx_above[x - 2]  },
                        est
                    ));

                    let avalue = add(slice[ox + 2], png_rev(
                        if x < 2 { 0 } else { ndx_line[x - 1] },
                        ndx_above[x + 1] ,
                        if x < 2 { 0 } else { ndx_above[x - 1]  },
                        est
                    ));

                    ndx_line.push(value);
                    ndx_line.push(avalue);

                    let val = (value as u32 * 0xffff_u32 / 0xff_u32) as u16;
                    let aval = (avalue as u32 * 0xffff_u32 / 0xff_u32) as u16;

                    graya_line.push((value as u16, avalue as u16));

                    line.push((
                        val,
                        val,
                        val,
                        aval
                    ));
                });
                raw_graya.push(graya_line);
            },
            ColorType::GRAY(Grayscale::G16) => {
                ndx_line = Vec::new();
                let mut gray_line = Vec::new();

                (0 .. width * 2).step_by(2).for_each(|ox| {
                    let x = ox;

                    let value_8_h = add(slice[ox + 1], png_rev(
                        if x < 2 { 0 } else { ndx_line[x - 2] },
                        ndx_above[x],
                        if x < 2 { 0 } else { ndx_above[x - 2]  },
                        est
                    ));

                    ndx_line.push(value_8_h);

                    let value_8_l = add(slice[ox + 2], png_rev(
                        if x < 2 { 0 } else { ndx_line[x - 1] },
                        ndx_above[x + 1],
                        if x < 2 { 0 } else { ndx_above[x - 1]  },
                        est
                    ));

                    ndx_line.push(value_8_l);

                    let val = ((value_8_h as u16) << 8) | value_8_l as u16;

                    gray_line.push(val);

                    line.push((
                        val,
                        val,
                        val,
                        0xffff_u16
                    ));
                });
                raw_gray.push(gray_line);
            },
            ColorType::GRAYA(Grayscale::G16) => {
                ndx_line = Vec::new();
                let mut graya_line = Vec::new();

                (0 .. width * 4).step_by(4).for_each(|ox| {
                    let x = ox;

                    let value_8_h = add(slice[ox + 1], png_rev(
                        if x < 4 { 0 } else { ndx_line[x - 4] },
                        ndx_above[x],
                        if x < 4 { 0 } else { ndx_above[x - 4]  },
                        est
                    ));

                    ndx_line.push(value_8_h);

                    let value_8_l = add(slice[ox + 2], png_rev(
                        if x < 4 { 0 } else { ndx_line[x - 3] },
                        ndx_above[x + 1],
                        if x < 4 { 0 } else { ndx_above[x - 3]  },
                        est
                    ));

                    ndx_line.push(value_8_l);

                    let avalue_8_h = add(slice[ox + 3], png_rev(
                        if x < 4 { 0 } else { ndx_line[x - 2] },
                        ndx_above[x + 2],
                        if x < 4 { 0 } else { ndx_above[x - 2]  },
                        est
                    ));

                    ndx_line.push(avalue_8_h);

                    let avalue_8_l = add(slice[ox + 4], png_rev(
                        if x < 4 { 0 } else { ndx_line[x - 1] },
                        ndx_above[x + 3],
                        if x < 4 { 0 } else { ndx_above[x - 1]  },
                        est
                    ));

                    ndx_line.push(avalue_8_l);

                    let val = ((value_8_h as u16) << 8) | value_8_l as u16;
                    let aval = ((avalue_8_h as u16) << 8) | avalue_8_l as u16;

                    graya_line.push((val, aval));

                    line.push((
                        val,
                        val,
                        val,
                        aval
                    ));
                });
                raw_graya.push(graya_line);
            },
            ColorType::GRAYA(Grayscale::G4) |
            ColorType::GRAYA(Grayscale::G2) |
            ColorType::GRAYA(Grayscale::G1) =>
                return Err("forbidden by specification".to_string()),
            ColorType::GRAY(Grayscale::G4) |
            ColorType::GRAY(Grayscale::G2) |
            ColorType::GRAY(Grayscale::G1) => {
                let line_width = match color_type {
                    ColorType::GRAY(Grayscale::G4) => (width + 1) / 2,
                    ColorType::GRAY(Grayscale::G2) => (width + 3) / 4,
                    ColorType::GRAY(Grayscale::G1) => (width + 7) / 8,
                    _ => panic!("unpack_idat bad case")
                };

                line = Vec::new();
                ndx_line = Vec::new();
                ndx_line_raw = Vec::new();
                let mut gray_line = Vec::new();

                let gray_status = (0 .. line_width).map(|ox| -> Result<(), String> {
                    let x = ox;
                    let ndx = add(slice[ox + 1], png_rev(
                            if x < 1 { 0 } else { ndx_line[x - 1]},
                            ndx_above[x],
                            if x < 1 { 0 } else { ndx_above[x - 1]},
                            est
                        )
                    );

                    let (unpacked, top) = match color_type {
                        ColorType::GRAY(Grayscale::G4) => {
                            (vec![
                                (ndx >> 4) & 0xf,
                                ndx & 0xf
                            ], 2)
                        },
                        ColorType::GRAY(Grayscale::G2) => {
                            (vec![
                                (ndx >> 6) & 3,
                                (ndx >> 4) & 3,
                                (ndx >> 2) & 3,
                                ndx & 3,
                            ], 4)
                        },
                        ColorType::GRAY(Grayscale::G1) => {
                            (vec![
                                (ndx >> 7) & 1,
                                (ndx >> 6) & 1,
                                (ndx >> 5) & 1,
                                (ndx >> 4) & 1,
                                (ndx >> 3) & 1,
                                (ndx >> 2) & 1,
                                (ndx >> 1) & 1,
                                ndx & 1,
                            ], 8)
                        },
                        _ => panic!("internal error @ unpack_idat / grayscale")
                    };

                    (0 .. top).for_each(|pos| {
                        if line.len() >= width {
                            return
                        }

                        let ndx1 = unpacked[pos];

                        let max = match color_type {
                            ColorType::GRAY(Grayscale::G4) => 0xf,
                            ColorType::GRAY(Grayscale::G2) => 3,
                            ColorType::GRAY(Grayscale::G1) => 1,
                            _ => panic!("unpack_idat internal error")
                        };

                        let val = (ndx1 as u32 * 0xffff_u32 / max as u32) as u16;

                        let pix: RGBA16 = (
                            val,
                            val,
                            val,
                            0xffff_u16
                        );

                        line.push(pix);
                        ndx_line_raw.push(ndx1);
                        gray_line.push(ndx1 as u16);
                    });
                    ndx_line.push(ndx);
                    Ok(())
                }).find(|x| x.is_err());
                if let Some(e) = gray_status {
                    return e;
                }
                raw_gray.push(gray_line);
            },
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
        ColorType::NDXA(p) => ImageData::NDXA(vec![raw_ndx], pal.to_vec(), p),
        ColorType::NDX(p) => {
            let pal_rgb: Vec<RGB> = pal.iter().map(|pal| {
                (
                    pal.0,
                    pal.1,
                    pal.2
                )
            }).collect();
            ImageData::NDX(vec![raw_ndx], pal_rgb, p)
        },
        ColorType::GRAYA(g) => ImageData::GRAYA(vec![raw_graya], g),
        ColorType::GRAY(g) => ImageData::GRAY(vec![raw_gray], g),
    };

    let rem = slice_len * height;

    assert!(rem <= raw.len());// XXX sth strange

    let rest = raw[rem ..].to_vec();

    match status {
        Some(Err(e)) => Err(e),
        Some(Ok(_)) => Err("internal unpack_idat error".to_string()),
        None => Ok((res, raw_data, rest))
    }
}

/// Decode PNG. For explanations see [read_png]. Input: `&[u8]`.
pub fn read_png_u8(buf: &[u8]) -> Result<Image, String> {
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

    let mut ps = 0xff_usize;

    let mut adam_7 = false;

    loop {
        let chunk = get_chunk(&buf[offs ..])?;

        offs += chunk.1.len() + 12_usize;

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
            let pack = chunk.1[10];
            let filter = chunk.1[11];
            let ilace = chunk.1[12];

            if color == 3 {
                match depth {
                    1 | 2 | 4 | 8 => (),
                    d => return Err(format!("unsupported color depth for indexed mode: {d}"))
                }
            }
            else if color == 2 || color == 6 {
                match depth {
                    8 | 16 => (),
                    d => return Err(format!("unsupported color depth for RGB(a) mode: {d}"))
                }
            }

            match color {
                6 => if depth == 8 { color_type = ColorType::RGBA } else { color_type = ColorType::RGBA16 },
                2 => if depth == 8 { color_type = ColorType::RGB } else { color_type = ColorType::RGB16 },
                3 => match depth {
                        1 => color_type = ColorType::NDX(Palette::P1),
                        2 => color_type = ColorType::NDX(Palette::P2),
                        4 => color_type = ColorType::NDX(Palette::P4),
                        8 => color_type = ColorType::NDX(Palette::P8),
                        d => return Err(format!("bad palette depth: {d}"))
                    },
                0 => match depth {
                        1 => color_type = ColorType::GRAY(Grayscale::G1),
                        2 => color_type = ColorType::GRAY(Grayscale::G2),
                        4 => color_type = ColorType::GRAY(Grayscale::G4),
                        8 => color_type = ColorType::GRAY(Grayscale::G8),
                        16 => color_type = ColorType::GRAY(Grayscale::G16),
                        d => return Err(format!("depth {d} for GRAY not supported"))
                    },
                4 => match depth {
                        8 => color_type = ColorType::GRAYA(Grayscale::G8),
                        16 => color_type = ColorType::GRAYA(Grayscale::G16),
                        d => return Err(format!("depth {d} for GRAYA not supported"))
                    },
                c => return Err(format!("color type {c} not supported"))
            }

            if pack != 0 {
                return Err(format!("unsupported compression {pack}"));
            }

            if filter != 0 {
                return Err(format!("unsupported filter {filter}"));
            }

            if ilace != 0 && ilace != 1 {
                return Err("bad interlacing mode".to_string())
            }
            else {
                adam_7 = ilace == 1;
            }
        }

        if prev_idat && chunk.0 != "IDAT" {
            let mut deco = ZlibDecoder::new(&unpacked[..]);
            let mut unpacked: Vec<u8> = Vec::new();

            if let Err(e) = deco.read_to_end(&mut unpacked) {
                return Err(format!("zlib IDAT: {e:?}"))
            }

            if !adam_7 {
                let (d, r, rest) = unpack_idat(width, height, &unpacked[..], color_type, &pal)?;

                if !rest.is_empty() {
                    return Err("IDAT buffer overflow".to_string())
                }

                data = d;
                raw = Some(r);
            }
            else { // Adam7
                let mut d = vec![vec![(0, 0, 0, 0); width]; height];
                let lo = 1;
                let hi = 7;
                let mut cur: Vec<u8> = unpacked;

                let mut map: Vec<Vec<usize>> = vec![vec![0_usize; width]; height];
                let mut slices = vec![0_usize; 8];

                (0 .. height).for_each(|y| {
                    (0 .. width).for_each(|x| {
                        let ax = x & 7;
                        let ay = y & 7;
                        map[y][x] = ADAM_7[ay * 8 + ax];
                        slices[ADAM_7[ay * 8 + ax] - 1] += 1;
                    });
                });

                let mut a = lo;

                let mut data_rgba: Vec<Vec<RGBA>> = Vec::new();
                let mut data_rgb: Vec<Vec<RGB>> = Vec::new();

                let mut data_rgba16: Vec<Vec<RGBA16>> = Vec::new();
                let mut data_rgb16: Vec<Vec<RGB16>> = Vec::new();

                let mut data_ndx: Vec<Vec<NDX>> = Vec::new();

                let mut data_gs: Vec<Vec<u16>> = Vec::new();
                let mut data_gsa: Vec<Vec<(u16, u16)>> = Vec::new();

                match color_type {
                    ColorType::RGBA => data_rgba = vec![vec![(0, 0, 0, 0); width]; height],
                    ColorType::RGB => data_rgb = vec![vec![(0, 0, 0); width]; height],
                    ColorType::RGBA16 => data_rgba16 = vec![vec![(0, 0, 0, 0); width]; height],
                    ColorType::RGB16 => data_rgb16 = vec![vec![(0, 0, 0); width]; height],
                    ColorType::NDXA(_) | ColorType::NDX(_) => data_ndx = vec![vec![0; width]; height],
                    ColorType::GRAY(_) => data_gs = vec![vec![0; width]; height],
                    ColorType::GRAYA(_) => data_gsa = vec![vec![(0, 0); width]; height],
                };

                while a <= hi {// TODO .map + result grep
                    let mut w = 0;
                    let mut h = 0;

                    (0 .. height).for_each(|y| {
                        let mut cur_w = 0;
                        (0 .. width).for_each(|x| {
                            if map[y][x] == a {
                                cur_w += 1;
                            }
                        });
                        assert!(cur_w == 0 || w == 0 || cur_w == w);
                        if cur_w > 0 {
                            w = cur_w;
                            h += 1;
                        }
                    });

                    if slices[a - 1] > 0 {// NOTE edge case
                        let (ds, r, rest) = unpack_idat(w, h, &cur[..], color_type, &pal)?;
                        cur = rest;

                        let mut sy = 0;
                        let mut sx = 0;

                        (0 .. height).for_each(|y| {
                            sx = 0;
                            let mut any = false;
                            (0 .. width).for_each(|x| {
                                if map[y][x] == a {
                                    d[y][x] = ds[sy][sx];

                                    match &r {
                                        ImageData::RGBA(rd) => data_rgba[y][x] = rd[0][sy][sx],
                                        ImageData::RGB(rd) => data_rgb[y][x] = rd[0][sy][sx],
                                        ImageData::RGBA16(rd) => data_rgba16[y][x] = rd[0][sy][sx],
                                        ImageData::RGB16(rd) => data_rgb16[y][x] = rd[0][sy][sx],
                                        ImageData::NDXA(rd, _, _) => data_ndx[y][x] = rd[0][sy][sx],
                                        ImageData::NDX(rd, _, _) => data_ndx[y][x] = rd[0][sy][sx],
                                        ImageData::GRAY(rd, _) => data_gs[y][x] = rd[0][sy][sx],
                                        ImageData::GRAYA(rd, _) => data_gsa[y][x] = rd[0][sy][sx],
                                    };

                                    sx += 1;
                                    any = true;
                                }
                            });
                            if any {
                                sy += 1;
                            }
                        });
                    }

                    a += 1;
                }

                data = d;

                raw = Some(match color_type {
                    ColorType::RGBA => ImageData::RGBA(vec![data_rgba]),
                    ColorType::RGB => ImageData::RGB(vec![data_rgb]),
                    ColorType::RGBA16 => ImageData::RGBA16(vec![data_rgba16]),
                    ColorType::RGB16 => ImageData::RGB16(vec![data_rgb16]),
                    ColorType::NDX(d) => ImageData::NDX(vec![data_ndx], pal.iter().map(|c| {
                        (c.0, c.1, c.2)
                    }).collect(), d),
                    ColorType::NDXA(d) => ImageData::NDXA(vec![data_ndx], pal.clone(), d),
                    ColorType::GRAY(g) => ImageData::GRAY(vec![data_gs], g),
                    ColorType::GRAYA(g) => ImageData::GRAYA(vec![data_gsa], g),
                });

                if !cur.is_empty() {
                    return Err("IDAT buffer overflow".to_string())
                }
            }
        }

        if chunk.0 == "IDAT" {
            unpacked.extend(chunk.1);
            prev_idat = true;
        }

        if chunk.0 == "PLTE" {
            match color_type {
                ColorType::NDXA(_) | ColorType::NDX(_) => {
                    ps = chunk.1.len() / 3;

                    (0 .. ps).for_each(|pndx| {
                        pal[pndx].0 = chunk.1[pndx * 3];
                        pal[pndx].1 = chunk.1[pndx * 3 + 1];
                        pal[pndx].2 = chunk.1[pndx * 3 + 2];
                    });
                    // NOTE just ignore
                }
                _ => ()
            }
        }

        if chunk.0 == "tRNS" {
            if let ColorType::NDX(p) = color_type {
                (0 .. chunk.1.len()).for_each(|pndx| {
                    pal[pndx].3 = chunk.1[pndx];
                });

                color_type = ColorType::NDXA(p); // NOTE upgrade
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

        if chunk.0 == "IEND" {
            break
        }
    }

    match color_type {
        ColorType::NDXA(_) | ColorType::NDX(_) => { // XXX trim palette
            let new_raw = match raw.unwrap() {
                ImageData::NDXA(d, p, m) =>
                    ImageData::NDXA(d, p[0 .. ps].to_vec(), m),
                ImageData::NDX(d, p, m) =>
                    ImageData::NDX(d, p[0 .. ps].to_vec(), m),
                _ => panic!("read_png_u8 internal error")
            };

            Ok(Image {
                width,
                height,
                color_type,
                data,
                meta,
                raw: new_raw
            })
        },
        _ =>
            Ok(Image {
                width,
                height,
                color_type,
                data,
                meta,
                raw: raw.unwrap()
            })
    }
}

/// Read png file.
///
/// # Arguments
///
/// * `fname` - input filename,
///
/// # Example
///
/// ```rust
/// use micro_png::*;
///
/// let image = read_png("fixtures/test.png").expect("can't load test.png");
///
/// println!("{} x {}", image.width(), image.height());
///
/// let data = image.data();
///
/// (0 .. image.height()).for_each(|y| {
///   (0 .. image.width()).for_each(|x| {
///     let _pixel = data[y][x]; // (u16, u16, u16, u16)
///   });
/// });
/// ```
pub fn read_png(fname: &str) -> Result<Image, String> {
    let mut input = match File::open(fname) {
        Ok(f) => f,
        Err(_) => return Err(format!("cannot open file {fname}"))
    };

    let input_len = match fs::metadata(fname) {
        Ok(meta) => meta.len() as usize,
        Err(_) => return Err(format!("cannot get metadata {fname}"))
    };

    let mut buf: Vec<u8> = vec![0; input_len];

    match input.read(&mut buf) {
        Ok(_) => (),
        Err(_) => return Err(format!("cannot read file {fname}"))
    };

    read_png_u8(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 133;
    const HEIGHT: usize = 193;

    fn image_rgba() -> (Vec<Vec<RGBA16>>, Vec<Vec<RGBA>>) {
        let w = WIDTH;
        let h = HEIGHT;

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
        let w = WIDTH;
        let h = HEIGHT;

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
        let w = WIDTH;
        let h = HEIGHT;

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
        let w = WIDTH;
        let h = HEIGHT;

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

    fn image_ndx(ps: usize) -> (Vec<Vec<RGBA16>>, // restored image
                                Vec<Vec<u8>>, // @ 0: pal ndx for writer,
                                Vec<RGB>) { // palette
        assert!(ps == 2 || ps == 4 || ps == 16 || ps == 256);

        let w = WIDTH;
        let h = HEIGHT;

        let mut pal = vec![(0, 0, 0); ps];

        let mut orig_img: Vec<Vec<RGBA16>> = Vec::new();
        let mut data_img: Vec<Vec<u8>> = Vec::new();

        (0 .. ps).for_each(|k| {
            let l = 255 * k / ps;
            pal[k].0 = l as u8;
            pal[k].1 = (64 + l) as u8;
            pal[k].2 = (128 + l) as u8;
        });

        (0 .. h).for_each(|y| {
            let mut orig_line: Vec<RGBA16> = Vec::new();
            let mut data_line: Vec<u8> = Vec::new();

            (0 .. w).for_each(|x| {
                let pndx = ((x + y) % ps as usize) as u8 as usize;
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

    fn image_ndxa(ps: usize) -> (Vec<Vec<RGBA16>>, // restored image
                                 Vec<Vec<u8>>, // @ 0: pal ndx for writer,
                                 Vec<RGBA>) { // palette
        assert!(ps == 2 || ps == 4 || ps == 16 || ps == 256);

        let w = WIDTH;
        let h = HEIGHT;

        let mut pal = vec![(0, 0, 0, 0); ps];

        let mut orig_img: Vec<Vec<RGBA16>> = Vec::new();
        let mut data_img: Vec<Vec<u8>> = Vec::new();

        (0 .. ps).for_each(|k| {
            let l = 255 * k / ps;
            pal[k].0 = l as u8;
            pal[k].1 = (64 + l) as u8;
            pal[k].2 = (128 + l) as u8;
            pal[k].3 = (255 - l) as u8;
        });

        (0 .. h).for_each(|y| {
            let mut orig_line: Vec<RGBA16> = Vec::new();
            let mut data_line: Vec<u8> = Vec::new();

            (0 .. w).for_each(|x| {
                let pndx = ((x + y) % ps as usize) as u8 as usize;
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

    fn image_gs(depth: usize) -> (Vec<Vec<RGBA16>>, // restored image
                                  Vec<Vec<u16>>) { // data
        assert!(depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16);

        let s = 1 << depth;

        let w = WIDTH;
        let h = HEIGHT;

        let mut orig_img: Vec<Vec<RGBA16>> = Vec::new();
        let mut data_img: Vec<Vec<u16>> = Vec::new();

        (0 .. h).for_each(|y| {
            let mut orig_line: Vec<RGBA16> = Vec::new();
            let mut data_line: Vec<u16> = Vec::new();

            (0 .. w).for_each(|x| {
                let ndx = ((x + y) * s / 0xff) % s;
                let org = (ndx as u32 * 0xffff_u32 / (s - 1) as u32) as u16;

                data_line.push(ndx as u16);
                orig_line.push((
                    org,
                    org,
                    org,
                    0xffff_u16
                ));
            });
            orig_img.push(orig_line);
            data_img.push(data_line);
        });

        (
            orig_img,
            data_img
        )
    }

    fn image_gsa(depth: usize) -> (Vec<Vec<RGBA16>>, // restored image
                                   Vec<Vec<(u16, u16)>>) { // data
        assert!(depth == 8 || depth == 16);

        let s = 1 << depth;

        let w = WIDTH;
        let h = HEIGHT;

        let mut orig_img: Vec<Vec<RGBA16>> = Vec::new();
        let mut data_img: Vec<Vec<(u16, u16)>> = Vec::new();

        (0 .. h).for_each(|y| {
            let mut orig_line: Vec<RGBA16> = Vec::new();
            let mut data_line: Vec<(u16, u16)> = Vec::new();

            (0 .. w).for_each(|x| {
                let ndx = ((x + y) * s / 0xff) % s;
                let org = (ndx as u32 * 0xffff_u32 / (s - 1) as u32) as u16;

                let andx = (x as i32 - y as i32) as usize % s;
                let aorg = (andx as u32 * 0xffff_u32 / (s - 1) as u32) as u16;

                data_line.push((ndx as u16, andx as u16));
                orig_line.push((
                    org,
                    org,
                    org,
                    aorg
                ));
            });
            orig_img.push(orig_line);
            data_img.push(data_line);
        });

        (
            orig_img,
            data_img
        )
    }

    #[test]
    pub fn test_rgba() {
        let (orig, image) = image_rgba();

        let types = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgba_{est:?}.png");
            let fname_a7 = format!("tmp/rgba_{est:?}_a7.png");

            write_apng(&fname,
                ImageData::RGBA(vec![image.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            write_apng(&fname_a7,
                ImageData::RGBA(vec![image.clone()]),
                Some(*est),
                None,
                true
            ).unwrap();

            let back = read_png(&fname).unwrap();
            let back_7 = read_png(&fname_a7).unwrap();

            assert_eq!(back.width, WIDTH);
            assert_eq!(back.height, HEIGHT);
            assert_eq!(back.color_type, ColorType::RGBA);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGBA(vec![image.clone()]));

            assert_eq!(back_7.width, WIDTH);
            assert_eq!(back_7.height, HEIGHT);
            assert_eq!(back_7.color_type, ColorType::RGBA);
            assert_eq!(back_7.data, orig);
            assert_eq!(back_7.raw, ImageData::RGBA(vec![image.clone()]));
        });
    }

    #[test]
    pub fn test_rgba_16() {
        let (orig, image) = image_rgba_16();

        let types = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgba_16_{est:?}.png");
            let fname_a7 = format!("tmp/rgba_16_{est:?}_a7.png");

            write_apng(&fname,
                ImageData::RGBA16(vec![image.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            write_apng(&fname_a7,
                ImageData::RGBA16(vec![image.clone()]),
                Some(*est),
                None,
                true
            ).unwrap();

            let back = read_png(&fname).unwrap();
            let back_7 = read_png(&fname_a7).unwrap();

            assert_eq!(back.width, WIDTH);
            assert_eq!(back.height, HEIGHT);
            assert_eq!(back.color_type, ColorType::RGBA16);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGBA16(vec![image.clone()]));

            assert_eq!(back_7.width, WIDTH);
            assert_eq!(back_7.height, HEIGHT);
            assert_eq!(back_7.color_type, ColorType::RGBA16);
            assert_eq!(back_7.data, orig);
            assert_eq!(back_7.raw, ImageData::RGBA16(vec![image.clone()]));
        });
    }

    #[test]
    pub fn test_rgb() {
        let (orig, data) = image_rgb();

        let types = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgb_{est:?}.png");
            let fname_a7 = format!("tmp/rgb_{est:?}_a7.png");

            write_apng(&fname,
                ImageData::RGB(vec![data.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            write_apng(&fname_a7,
                ImageData::RGB(vec![data.clone()]),
                Some(*est),
                None,
                true
            ).unwrap();

            let back = read_png(&fname).unwrap();
            let back_7 = read_png(&fname_a7).unwrap();

            assert_eq!(back.width, WIDTH);
            assert_eq!(back.height, HEIGHT);
            assert_eq!(back.color_type, ColorType::RGB);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGB(vec![data.clone()]));

            assert_eq!(back_7.width, WIDTH);
            assert_eq!(back_7.height, HEIGHT);
            assert_eq!(back_7.color_type, ColorType::RGB);
            assert_eq!(back_7.data, orig);
            assert_eq!(back_7.raw, ImageData::RGB(vec![data.clone()]));
        });
    }

    #[test]
    pub fn test_rgb_16() {
        let (orig, data) = image_rgb_16();

        let types = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        types.iter().for_each(|est| {
            println!("{est:?}");

            let fname = format!("tmp/rgb_16_{est:?}.png");
            let fname_a7 = format!("tmp/rgb_16_{est:?}_a7.png");

            write_apng(&fname,
                ImageData::RGB16(vec![data.clone()]),
                Some(*est),
                None,
                false
            ).unwrap();

            write_apng(&fname_a7,
                ImageData::RGB16(vec![data.clone()]),
                Some(*est),
                None,
                true
            ).unwrap();

            let back = read_png(&fname).unwrap();
            let back_7 = read_png(&fname_a7).unwrap();

            assert_eq!(back.width, WIDTH);
            assert_eq!(back.height, HEIGHT);
            assert_eq!(back.color_type, ColorType::RGB16);
            assert_eq!(back.data, orig);
            assert_eq!(back.raw, ImageData::RGB16(vec![data.clone()]));

            assert_eq!(back_7.width, WIDTH);
            assert_eq!(back_7.height, HEIGHT);
            assert_eq!(back_7.color_type, ColorType::RGB16);
            assert_eq!(back_7.data, orig);
            assert_eq!(back_7.raw, ImageData::RGB16(vec![data.clone()]));
        });
    }

    #[test]
    pub fn test_ndx() {
        let presets = vec![
            (8, Palette::P8),
            (4, Palette::P4),
            (2, Palette::P2),
            (1, Palette::P1),
        ];

        let filters = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        presets.iter().for_each(|(ps, pt)| {
            filters.iter().for_each(|ft| {
                let nc = 1 << ps;
                println!("{ps} {nc} {pt:?} {ft:?}");

                let fname = format!("tmp/ndx_{ft:?}_{ps}.png");
                let fname_a7 = format!("tmp/ndx_{ft:?}_{ps}_a7.png");

                let (orig, data, pal) = image_ndx(nc);

                write_apng(&fname,
                    ImageData::NDX(vec![data.clone()], pal.clone(), *pt),
                    Some(*ft),
                    None,
                    false
                ).unwrap();

                write_apng(&fname_a7,
                    ImageData::NDX(vec![data.clone()], pal.clone(), *pt),
                    Some(*ft),
                    None,
                    true
                ).unwrap();

                let back = read_png(&fname).unwrap();
                let back_7 = read_png(&fname_a7).unwrap();

                assert_eq!(back.width, WIDTH);
                assert_eq!(back.height, HEIGHT);
                assert_eq!(back.color_type, ColorType::NDX(*pt));
                assert_eq!(back.data, orig);
                assert_eq!(back.raw, ImageData::NDX(vec![data.clone()], pal.clone(), *pt));

                assert_eq!(back_7.width, WIDTH);
                assert_eq!(back_7.height, HEIGHT);
                assert_eq!(back_7.color_type, ColorType::NDX(*pt));
                assert_eq!(back_7.data, orig);
                assert_eq!(back_7.raw, ImageData::NDX(vec![data.clone()], pal.clone(), *pt));
            });
        });
    }

    #[test]
    pub fn test_ndxa() {
        let presets = vec![
            (8, Palette::P8),
            (4, Palette::P4),
            (2, Palette::P2),
            (1, Palette::P1),
        ];

        let filters = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        presets.iter().for_each(|(ps, pt)| {
            filters.iter().for_each(|ft| {
                let nc = 1 << ps;
                println!("{ps} {nc} {ft:?}");

                let fname = format!("tmp/ndxa_{ft:?}_{ps}.png");
                let fname_a7 = format!("tmp/ndxa_{ft:?}_{ps}_a7.png");

                let (orig, data, pal) = image_ndxa(nc);

                write_apng(&fname,
                    ImageData::NDXA(vec![data.clone()], pal.clone(), *pt),
                    Some(*ft),
                    None,
                    false
                ).unwrap();

                write_apng(&fname_a7,
                    ImageData::NDXA(vec![data.clone()], pal.clone(), *pt),
                    Some(*ft),
                    None,
                    true
                ).unwrap();

                let back = read_png(&fname).unwrap();
                let back_7 = read_png(&fname_a7).unwrap();

                assert_eq!(back.width, WIDTH);
                assert_eq!(back.height, HEIGHT);
                assert_eq!(back.color_type, ColorType::NDXA(*pt));
                assert_eq!(back.data, orig);
                assert_eq!(back.raw, ImageData::NDXA(vec![data.clone()], pal.clone(), *pt));

                assert_eq!(back_7.width, WIDTH);
                assert_eq!(back_7.height, HEIGHT);
                assert_eq!(back_7.color_type, ColorType::NDXA(*pt));
                assert_eq!(back_7.data, orig);
                assert_eq!(back_7.raw, ImageData::NDXA(vec![data.clone()], pal.clone(), *pt));
            });
        });
    }

    #[test]
    pub fn test_gs() {
        let presets = vec![
            (16, Grayscale::G16),
            (8,  Grayscale::G8),
            (4,  Grayscale::G4),
            (2,  Grayscale::G2),
            (1,  Grayscale::G1),
        ];

        let filters = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        presets.iter().for_each(|(gw, gt)| {
            filters.iter().for_each(|ft| {
                println!("{gw} {gt:?} {ft:?}");

                let fname = format!("tmp/gs_{ft:?}_{gw}.png");
                let fname_a7 = format!("tmp/gs_{ft:?}_{gw}_a7.png");

                let (orig, data) = image_gs(*gw);

                write_apng(&fname,
                    ImageData::GRAY(vec![data.clone()], *gt),
                    Some(*ft),
                    None,
                    false
                ).unwrap();

                write_apng(&fname_a7,
                    ImageData::GRAY(vec![data.clone()], *gt),
                    Some(*ft),
                    None,
                    true
                ).unwrap();

                let back = read_png(&fname).unwrap();
                let back_7 = read_png(&fname_a7).unwrap();

                assert_eq!(back.width, WIDTH);
                assert_eq!(back.height, HEIGHT);
                assert_eq!(back.color_type, ColorType::GRAY(*gt));
                assert_eq!(back.data, orig);
                assert_eq!(back.raw, ImageData::GRAY(vec![data.clone()], *gt));

                assert_eq!(back_7.width, WIDTH);
                assert_eq!(back_7.height, HEIGHT);
                assert_eq!(back_7.color_type, ColorType::GRAY(*gt));
                assert_eq!(back_7.data, orig);
                assert_eq!(back_7.raw, ImageData::GRAY(vec![data.clone()], *gt));
            });
        });
    }

    #[test]
    pub fn test_gsa() {
        let presets = vec![
            (16, Grayscale::G16),
            (8,  Grayscale::G8),
        ];

        let filters = vec![
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth
        ];

        presets.iter().for_each(|(gw, gt)| {
            filters.iter().for_each(|ft| {
                println!("{gw} {gt:?} {ft:?}");

                let fname = format!("tmp/gsa_{ft:?}_{gw}.png");
                let fname_a7 = format!("tmp/gsa_{ft:?}_{gw}_a7.png");

                let (orig, data) = image_gsa(*gw);

                write_apng(&fname,
                    ImageData::GRAYA(vec![data.clone()], *gt),
                    Some(*ft),
                    None,
                    false
                ).unwrap();

                write_apng(&fname_a7,
                    ImageData::GRAYA(vec![data.clone()], *gt),
                    Some(*ft),
                    None,
                    true
                ).unwrap();

                let back = read_png(&fname).unwrap();
                let back_7 = read_png(&fname_a7).unwrap();

                assert_eq!(back.width, WIDTH);
                assert_eq!(back.height, HEIGHT);
                assert_eq!(back.color_type, ColorType::GRAYA(*gt));
                assert_eq!(back.data, orig);
                assert_eq!(back.raw, ImageData::GRAYA(vec![data.clone()], *gt));

                assert_eq!(back_7.width, WIDTH);
                assert_eq!(back_7.height, HEIGHT);
                assert_eq!(back_7.color_type, ColorType::GRAYA(*gt));
                assert_eq!(back_7.data, orig);
                assert_eq!(back_7.raw, ImageData::GRAYA(vec![data.clone()], *gt));
            });
        });
    }

    #[test]
    pub fn test_meta() {
        let data: Vec<Vec<RGBA>> = vec![
            vec![(255, 0, 0, 255), (0, 0, 0, 255)],// the 1st line
            vec![(0, 0, 0, 255), (255, 0, 0, 255)],// the 2nd line
        ];

        build_apng(
            APNGBuilder::new("tmp/meta.png", ImageData::RGBA(vec![data]))
                .set_meta("Comment", "test comment")
                .set_zmeta("Author", "test author")
        ).expect("can't write tmp/meta.png");

        let back = read_png("tmp/meta.png").unwrap();
        let meta = back.meta();

        assert_eq!(meta["Comment"], "test comment");
        assert_eq!(meta["Author"], "test author");
    }

    #[test]
    #[should_panic(expected = "the image is empty")]
    fn test_no_frames() {
        build_apng(
            APNGBuilder::new("tmp/panic-0.png", ImageData::RGBA(vec![]))
        ).unwrap();
    }

    #[test]
    #[should_panic(expected = "the height is zero")]
    fn test_zero_h() {
        build_apng(
            APNGBuilder::new("tmp/panic-0.png", ImageData::RGBA(vec![vec![]]))
        ).unwrap();
    }

    #[test]
    #[should_panic(expected = "the width is zero")]
    fn test_zero_w() {
        build_apng(
            APNGBuilder::new("tmp/panic-0.png", ImageData::RGBA(vec![vec![vec![]]]))
        ).unwrap();
    }

    #[test]
    #[should_panic(expected = "bad frame #1 height: 2 instead of 1")]
    fn test_bad_h() {
        build_apng(
            APNGBuilder::new("tmp/panic-0.png", ImageData::RGBA(vec![
                    vec![
                        vec![(0, 0, 0, 0)]
                    ],
                    vec![
                        vec![(0, 0, 0, 0)],
                        vec![(0, 0, 0, 0)]
                    ]
            ]))
        ).unwrap();
    }

    #[test]
    #[should_panic(expected = "bad line width #0 at frame #1 height: 2 instead of 1")]
    fn test_bad_w() {
        build_apng(
            APNGBuilder::new("tmp/panic-0.png", ImageData::RGBA(vec![
                    vec![
                        vec![(0, 0, 0, 0)],
                        vec![(0, 0, 0, 0)]
                    ],
                    vec![
                        vec![(0, 0, 0, 0), (0, 0, 0, 0)],
                        vec![(0, 0, 0, 0)],
                    ]
            ]))
        ).unwrap();
    }
}
