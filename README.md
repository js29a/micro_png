usage
=====

```rust
use micro_png::{read_png, write_apng, Image, ImageData};

fn main() {
    // load an image
    let image = read_png("test.png").expect("can't load test.png");

    println!("{} x {}", image.width(), image.height());

    let data = image.data();

    (0 .. image.height()).for_each(|y| {
      (0 .. image.width()).for_each(|x| {
        let pixel = data[y][x]; // (R, G, B, A) as (u16, u16, u16, u16)
      });
    });

    // now write it back as one-frame image

    write_apng("back.png",
        &ImageData::RGBA16(vec![data]),
        None ,// automatically select filtering
        None, // no progress callback
        false // no Adam-7
    ).expect("can't save back.png");
}
```

supported formats
=================

| enum variant      |                                     |
|-------------------|-------------------------------------|
| ImageData::RGB    | 8-bit RGB without alpha             |
| ImageData::RGBA   | 8-bit RGB with alpha                |
| ImageData::RGB16  | 16-bit RGB without alpha            |
| ImageData::RGBA16 | 16-bit RGB with alpha               |
| ImageData::NDX    | 8-bit indexed palette without alpha |
| ImageData::NDXA   | 8-bit indexed palette with alpha    |

todo
====

- grayscale I/O,
- Adam-7 input,
- 1, 2, 4 palette bits,
- APNG input,

