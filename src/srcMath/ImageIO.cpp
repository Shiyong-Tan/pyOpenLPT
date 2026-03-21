#include "ImageIO.h"

#include <filesystem>


ImageIO::ImageIO (const ImageIO& img)
    : _n_row(img._n_row), _n_col(img._n_col), _bits_per_sample(img._bits_per_sample), _n_channel(img._n_channel), _is_tiled(img._is_tiled), _tile_height0(img._tile_height0), _tile_width0(img._tile_width0), _img_orientation(img._img_orientation), _img_id(img._img_id), _img_path(img._img_path)
{}

ImageIO::ImageIO (std::string folder_path, std::string file_img_path)
{
    loadImgPath(folder_path, file_img_path);
}

void ImageIO::init ()
{
    _n_row = 0;
    _n_col = 0;
    _bits_per_sample = 0;
    _n_channel = 0;
    _is_tiled = 0;
    _tile_height0 = 0;
    _tile_width0 = 0;
    _img_orientation = ORIENTATION_TOPLEFT;
    _img_id = -1;
    _img_path.clear();
}


void ImageIO::loadImgPath (std::string folder_path, std::string file_img_path)
{
    namespace fs = std::filesystem;

    fs::path base_dir;
    if (!folder_path.empty())
    {
        base_dir = fs::path(folder_path);
    }
    else
    {
        std::error_code ec;
        base_dir = fs::current_path(ec);
        if (ec)
        {
            base_dir.clear();
        }
    }

    fs::path list_path(file_img_path);
    if (!list_path.is_absolute())
    {
        list_path = base_dir / list_path;
    }
    list_path = list_path.lexically_normal();

    std::ifstream infile(list_path.string(), std::ios::in);

    if (!infile.is_open())
    {
        std::cerr << "ImageIO::LoadImgPath: Could not open image path file: "
                  << list_path.string() << std::endl;
        throw error_io;
    }

    // Initialize image path
    init();

    const fs::path list_dir = list_path.parent_path();
    const fs::path image_base_dir = base_dir.empty() ? list_dir : base_dir;

    std::string line;
    while (std::getline(infile, line)) 
    {
        // trim spaces/tabs and CR
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
            continue;
        }
        const size_t last = line.find_last_not_of(" \t\r\n");
        line = line.substr(first, last - first + 1);
        if (line.empty())
        {
            continue;
        }

        fs::path img_path(line);
        if (!img_path.is_absolute())
        {
            // Project convention: relative image paths are resolved from project root.
            // Here, folder_path is expected to be project/config root.
            img_path = image_base_dir / img_path;
        }
        _img_path.push_back(img_path.lexically_normal().string());
    }
    infile.close();

    if (_img_path.size() == 0)
    {
        std::cout << "There is no image path loaded." << std::endl;
    }
}


Image ImageIO::loadImg (int img_id)
{
    if (img_id >= int(_img_path.size()))
    {
        std::cerr << "Image id: " << img_id 
                  << " is larger than total number of image: " 
                  << _img_path.size()
                  << std::endl;
        throw error_size;
    }
    _img_id = img_id;
    std::string file = _img_path[img_id];
    
    TIFF* tif;
    if ((tif = TIFFOpen(file.c_str(), "r")) == NULL) 
    {
        std::cerr << "ImageIO::LoadImg: Could not open image!" << std::endl;
        throw error_io;
    }

    // check is the image is colorful
    _n_channel = 1;

    #ifdef DEBUG
    IMAGEIO_CHECK_CALL_DEBUG(TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &_n_channel));
    #endif
    if (_n_channel != 1)
    {
        std::cerr << "ImageIO::LoadImg: current version not supported for colorful image! " 
                  << "Line: " << __LINE__
                  << std::endl;
        throw error_io;
    }

    // check image size
    IMAGEIO_CHECK_CALL(TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &_n_row));
    IMAGEIO_CHECK_CALL(TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &_n_col));
    IMAGEIO_CHECK_CALL((_n_row > 0 && _n_col > 0));
    
    // load image bits
    IMAGEIO_CHECK_CALL(TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &_bits_per_sample));
    IMAGEIO_CHECK_CALL((_bits_per_sample==8 || _bits_per_sample==16 || _bits_per_sample==32 || _bits_per_sample==64));

    // check is the image is tiled or stripped
    _is_tiled = TIFFIsTiled(tif) != 0;
    bool is_read_scanline = false;
    if (_is_tiled)
    {
        IMAGEIO_CHECK_CALL(TIFFGetField(tif, TIFFTAG_TILELENGTH, &_tile_height0));
        IMAGEIO_CHECK_CALL(TIFFGetField(tif, TIFFTAG_TILEWIDTH, &_tile_width0));
        IMAGEIO_CHECK_CALL((_tile_height0>0 && _tile_height0<=TILE_MAX_HEIGHT && _tile_width0>0 && _tile_width0<=TILE_MAX_WIDTH));

        // For debug
        // if (_tile_height0 > TILE_MAX_HEIGHT || _tile_width0 > TILE_MAX_WIDTH || _tile_height0 <= 0 || _tile_width0 <= 0)
        // {
        //     std::cerr << "ImageIO::LoadImg: Image tile size is out of range! " 
        //               << "_tile_height0: " << _tile_height0 << " "
        //               << "_tile_width0: " << _tile_width0 << " "
        //               << "Line: " << __LINE__
        //               << std::endl;
        //     throw error_io;
        // }
    }
    else 
    {
        _tile_width0 = _n_col;

        if (_bits_per_sample <= 32)
        {
            is_read_scanline = true;
            _tile_height0 = 1; // read each scanline
        }
        else 
        {
            _tile_height0 = _n_row;
        }
    }
    
    // create buffer to store image data
    const size_t buffer_bytes_per_row = (_n_channel*_tile_width0*_bits_per_sample + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    const size_t buffer_size = _tile_height0 * buffer_bytes_per_row;
    if (buffer_size > MAX_TILE_SIZE)
    {
        std::cerr << "ImageIO::LoadImg: Image buffer size is out of range! " 
                  << "buffer_size: " << buffer_size << " "
                  << "Line: " << __LINE__
                  << std::endl;
        throw error_io;
    }
    uchar* buffer = new uchar[buffer_size]; 
    
    // check image orientation
    _img_orientation = ORIENTATION_TOPLEFT;
    #ifdef DEBUG
    IMAGEIO_CHECK_CALL_DEBUG(TIFFGetField(tif, TIFFTAG_ORIENTATION, &_img_orientation));
    #endif
    bool vert_flip = _img_orientation == ORIENTATION_BOTLEFT || _img_orientation == ORIENTATION_BOTRIGHT || _img_orientation == ORIENTATION_LEFTBOT || _img_orientation == ORIENTATION_RIGHTBOT;

    // initialize data array to convert buffer
    const size_t data_num_per_row = _n_channel * _tile_width0 * _tile_height0;
    uint8*  buffer_8 = nullptr;
    uint16* buffer_16 = nullptr;
    uint32* buffer_32 = nullptr;
    uint64* buffer_64 = nullptr;
    switch (_bits_per_sample)
    {
    case 8:
        buffer_8 = new uint8[data_num_per_row];
        buffer_16 = nullptr;
        buffer_32 = nullptr;
        buffer_64 = nullptr;
        break;
    case 16:
        buffer_8 = nullptr;
        buffer_16 = new uint16[data_num_per_row];
        buffer_32 = nullptr;
        buffer_64 = nullptr;
        break;
    case 32:
        buffer_8 = nullptr;
        buffer_16 = nullptr;
        buffer_32 = new uint32[data_num_per_row];
        buffer_64 = nullptr;
        break;
    case 64:
        buffer_8 = nullptr;
        buffer_16 = nullptr;
        buffer_32 = nullptr;
        buffer_64 = new uint64[data_num_per_row];
        break;
    default:
        std::cerr << "ImageIO::LoadImg: Image bits per sample is out of range! " 
                  << "_bits_per_sample: " << _bits_per_sample << " "
                  << "Line: " << __LINE__
                  << std::endl;
        break;
    }

    // read image data
    Image image(_n_row, _n_col, -1);
    int tile_id = 0;
    for (int row = 0; row < _n_row; row += _tile_height0)
    {
        int tile_height = std::min(_tile_height0, _n_row - row);
        const int img_row = vert_flip ? _n_row-1-row : row; 

        for (int col = 0; col < _n_col; col += _tile_width0, tile_id ++)
        {
            int tile_width = std::min(_tile_width0, _n_col - col);

            if (_is_tiled)
            {
                IMAGEIO_CHECK_CALL((TIFFReadEncodedTile(tif, tile_id, buffer, buffer_size) >= 0));
            }
            else if (is_read_scanline)
            {
                IMAGEIO_CHECK_CALL((TIFFReadScanline(tif, (uint32*)buffer, row) >= 0));
            }
            else
            {
                IMAGEIO_CHECK_CALL((TIFFReadEncodedStrip(tif, tile_id, buffer, buffer_size) >= 0));
            }

            switch (_bits_per_sample)
            {
            case 8:
                std::memcpy(buffer_8, buffer, buffer_size);
                for (int i = 0; i < tile_height; i ++)
                {
                    for (int j = 0; j < tile_width; j ++)
                    {
                        image(img_row+i, col+j) = (double) buffer_8[i*tile_width+j];
                    }
                }
                break;
            case 16:
                std::memcpy(buffer_16, buffer, buffer_size);
                for (int i = 0; i < tile_height; i ++)
                {
                    for (int j = 0; j < tile_width; j ++)
                    {
                        image(img_row+i, col+j) = (double) buffer_16[i*tile_width+j];
                    }
                }
                break;
            case 32:
                std::memcpy(buffer_32, buffer, buffer_size);
                for (int i = 0; i < tile_height; i ++)
                {
                    for (int j = 0; j < tile_width; j ++)
                    {
                        image(img_row+i, col+j) = (double) buffer_32[i*tile_width+j];
                    }
                }
                break;
            case 64:
                std::memcpy(buffer_64, buffer, buffer_size);
                for (int i = 0; i < tile_height; i ++)
                {
                    for (int j = 0; j < tile_width; j ++)
                    {
                        image(img_row+i, col+j) = (double) buffer_64[i*tile_width+j];
                    }
                }
                break;
            default:
                break;
            } 
        }
    }
    
    switch (_bits_per_sample)
    {
    case 8:
        delete[] buffer_8;
        break;
    case 16:
        delete[] buffer_16;
        break;
    case 32:
        delete[] buffer_32;
        break;
    case 64:
        delete[] buffer_64;
        break;
    default:
        break;
    }

    _TIFFfree(buffer);
    TIFFClose(tif);

    return image;
}


static inline uint8_t  clamp_u8 (double v)  { if (v<0) v=0; if (v>255) v=255; return (uint8_t)(v + 0.5); }
static inline uint16_t clamp_u16(double v)  { if (v<0) v=0; if (v>65535) v=65535; return (uint16_t)(v + 0.5); }
static inline uint32_t clamp_u32(double v)  { if (v<0) v=0; if (v>4294967295.0) v=4294967295.0; return (uint32_t)(v + 0.5); }
static inline uint64_t clamp_u64(long double v){ if (v<0) v=0; if (v>static_cast<long double>(UINT64_MAX)) v=static_cast<long double>(UINT64_MAX); return (uint64_t)(v + 0.5L); }

void ImageIO::saveImg (std::string save_path, Image const& image)
{
    // 与 setImgParam 一致的基本检查
    IMAGEIO_CHECK_CALL((_n_row>0 && _n_col>0 && _n_row==image.getDimRow() && _n_col==image.getDimCol()));

    TIFF* tif = TIFFOpen(save_path.c_str(), "w");
    REQUIRE(tif, ErrorCode::IOfailure, "ImageIO::saveImg: cannot open tiff for write: " + save_path);

    // 常用标签
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,        _n_col);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,       _n_row);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL,   _n_channel);        // 一般灰度=1
    TIFFSetField(tif, TIFFTAG_ORIENTATION,       _img_orientation);  // e.g. ORIENTATION_TOPLEFT
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,       PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,      PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,       COMPRESSION_NONE);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,     _bits_per_sample);

    const int bytes_per_sample = std::max(1, _bits_per_sample / 8);
    const int bytes_per_row    = _n_col * bytes_per_sample * std::max(1, _n_channel);
    const int rows_per_strip   = std::max(1, (1<<16) / std::max(1, bytes_per_row));
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rows_per_strip);

    switch (_bits_per_sample)
    {
    case 8:
    {
        uint8_t* buf = (uint8_t*) _TIFFmalloc(_n_row * _n_col * sizeof(uint8_t));
        if (!buf) { TIFFClose(tif); return; }
        for (int r = 0; r < _n_row; ++r)
            for (int c = 0; c < _n_col; ++c)
                buf[_n_col * r + c] = clamp_u8(image(r, c));

        for (int r = 0; r < _n_row; ++r)
            TIFFWriteScanline(tif, &buf[_n_col * r], r, 0); // 注意步长用 _n_col

        _TIFFfree(buf);
        break;
    }

    case 16:
    {
        uint16_t* buf = (uint16_t*) _TIFFmalloc(_n_row * _n_col * sizeof(uint16_t));
        if (!buf) { TIFFClose(tif); return; }
        for (int r = 0; r < _n_row; ++r)
            for (int c = 0; c < _n_col; ++c)
                buf[_n_col * r + c] = clamp_u16(image(r, c));

        for (int r = 0; r < _n_row; ++r)
            TIFFWriteScanline(tif, &buf[_n_col * r], r, 0);

        _TIFFfree(buf);
        break;
    }

    case 32:
    {
        // 这是 32 位无符号整型灰度；如需 float32，请改为 float* 并设 SAMPLEFORMAT=IEEEFP
        uint32_t* buf = (uint32_t*) _TIFFmalloc(_n_row * _n_col * sizeof(uint32_t));
        if (!buf) { TIFFClose(tif); return; }
        for (int r = 0; r < _n_row; ++r)
            for (int c = 0; c < _n_col; ++c)
                buf[_n_col * r + c] = clamp_u32(image(r, c));

        for (int r = 0; r < _n_row; ++r)
            TIFFWriteScanline(tif, &buf[_n_col * r], r, 0);

        _TIFFfree(buf);
        break;
    }

    case 64:
    {
        // 这是 64 位无符号整型灰度
        uint64_t* buf = (uint64_t*) _TIFFmalloc(_n_row * _n_col * sizeof(uint64_t));
        if (!buf) { TIFFClose(tif); return; }
        for (int r = 0; r < _n_row; ++r)
            for (int c = 0; c < _n_col; ++c)
                buf[_n_col * r + c] = clamp_u64((long double)image(r, c));

        for (int r = 0; r < _n_row; ++r)
            TIFFWriteScanline(tif, &buf[_n_col * r], r, 0);

        _TIFFfree(buf);
        break;
    }

    default:
        std::cerr << "ImageIO::saveImg: unsupported bits_per_sample = " << _bits_per_sample << std::endl;
        TIFFClose(tif);
        return;
    }

    TIFFClose(tif);
}

void ImageIO::setImgParam (ImageParam const& img_param)
{
    _n_row = img_param.n_row;
    _n_col = img_param.n_col;
    _bits_per_sample = img_param.bits_per_sample;
    _n_channel = img_param.n_channel;
    _img_orientation = img_param.img_orientation;
}


ImageParam ImageIO::getImgParam () const
{
    ImageParam img_param;
    img_param.n_row = _n_row;
    img_param.n_col = _n_col;
    img_param.bits_per_sample = _bits_per_sample;
    img_param.n_channel = _n_channel;
    img_param.img_orientation = _img_orientation;

    return img_param;
}



void Image::save(const std::string& path,
                 int bits_per_sample /*=8*/,
                 int n_channel       /*=1*/,
                 std::uint16_t orientation  /*=ORIENTATION_TOPLEFT*/) const
{
    ImageParam prm;
    prm.n_row           = this->getDimRow();
    prm.n_col           = this->getDimCol();
    prm.bits_per_sample = bits_per_sample;
    prm.n_channel       = n_channel;
    prm.img_orientation = orientation;

    ImageIO io;
    io.setImgParam(prm);
    io.saveImg(path, *this);
}
