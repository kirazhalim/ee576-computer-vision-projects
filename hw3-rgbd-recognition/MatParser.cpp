/*
 * Author: Abdulhalim Kiraz
 * course: ee576
 * project no: 3
 * file: matparser.cpp
 * description: reads scene ground truth boxes from a matlab v5 file.
 */

#include "MatParser.h"
#include "Project3_Params.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

// stores one matlab v5 tag.
struct MatTag {
    unsigned int type;
    unsigned int size;
    size_t dataPos;
    size_t nextPos;
};

// stores the header data of one matlab matrix.
struct MatMatrix {
    int cls;
    std::vector<int> dims;
    std::string name;
    size_t dataPos;
    size_t endPos;
};

// aligns mat blocks to 8 bytes.
int align8(int n) {
    return (n + 7) & ~7;
}

// maps a class name to its project id.
int matClassIndex(const std::string& className) {
    for (int i = 0; i < (int)CLASS_NAMES.size(); i++)
        if (CLASS_NAMES[i] == className)
            return i;
    return -1;
}

// keeps a rectangle inside image bounds.
cv::Rect matClampRect(const cv::Rect& r, int cols, int rows) {
    int x1 = std::max(0, r.x);
    int y1 = std::max(0, r.y);
    int x2 = std::min(cols, r.x + r.width);
    int y2 = std::min(rows, r.y + r.height);
    if (x2 <= x1 || y2 <= y1) return cv::Rect();
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// reads a binary file into memory.
std::vector<unsigned char> readFileBytes(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f.good()) return std::vector<unsigned char>();
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(n);
    f.read((char*)data.data(), n);
    return data;
}

// reads a little-endian unsigned integer.
unsigned int readU32(const std::vector<unsigned char>& data, size_t pos) {
    if (pos + 4 > data.size()) throw std::runtime_error("MAT parse error");
    return (unsigned int)data[pos] |
           ((unsigned int)data[pos + 1] << 8) |
           ((unsigned int)data[pos + 2] << 16) |
           ((unsigned int)data[pos + 3] << 24);
}

// reads a little-endian signed integer.
int readI32(const std::vector<unsigned char>& data, size_t pos) {
    return (int)readU32(data, pos);
}

// reads a matlab v5 data tag.
MatTag readTag(const std::vector<unsigned char>& data, size_t pos) {
    if (pos + 8 > data.size()) throw std::runtime_error("MAT tag error");

    unsigned int raw = readU32(data, pos);
    unsigned int smallType = raw & 0xffff;
    unsigned int smallSize = raw >> 16;

    if (smallSize > 0 && smallType > 0 && smallType <= 18) {
        MatTag tag;
        tag.type = smallType;
        tag.size = smallSize;
        tag.dataPos = pos + 4;
        tag.nextPos = pos + 8;
        return tag;
    }

    MatTag tag;
    tag.type = raw;
    tag.size = readU32(data, pos + 4);
    tag.dataPos = pos + 8;
    tag.nextPos = tag.dataPos + align8((int)tag.size);
    return tag;
}

// inflates the compressed mat block.
std::vector<unsigned char> inflateBytes(const unsigned char* ptr, size_t size) {
    std::vector<unsigned char> out;
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));

    if (inflateInit(&zs) != Z_OK)
        throw std::runtime_error("Could not initialize zlib");

    zs.next_in = const_cast<Bytef*>(ptr);
    zs.avail_in = (uInt)size;

    unsigned char buffer[32768];
    int ret = Z_OK;
    while (ret == Z_OK) {
        zs.next_out = buffer;
        zs.avail_out = sizeof(buffer);
        ret = inflate(&zs, 0);

        size_t have = sizeof(buffer) - zs.avail_out;
        out.insert(out.end(), buffer, buffer + have);
    }

    inflateEnd(&zs);
    if (ret != Z_STREAM_END)
        throw std::runtime_error("Could not inflate MAT contents");

    return out;
}

// reads the header of a matlab matrix.
MatMatrix readMatrixHeader(const std::vector<unsigned char>& data, size_t pos) {
    MatTag outer = readTag(data, pos);
    if (outer.type != 14)
        throw std::runtime_error("Expected miMATRIX");

    size_t cur = outer.dataPos;
    MatTag flags = readTag(data, cur);
    int cls = (int)(readU32(data, flags.dataPos) & 0xff);
    cur = flags.nextPos;

    MatTag dimsTag = readTag(data, cur);
    std::vector<int> dims;
    for (unsigned int i = 0; i < dimsTag.size; i += 4)
        dims.push_back(readI32(data, dimsTag.dataPos + i));
    cur = dimsTag.nextPos;

    MatTag nameTag = readTag(data, cur);
    std::string name;
    for (unsigned int i = 0; i < nameTag.size; i++)
        name.push_back((char)data[nameTag.dataPos + i]);
    cur = nameTag.nextPos;

    MatMatrix m;
    m.cls = cls;
    m.dims = dims;
    m.name = name;
    m.dataPos = cur;
    m.endPos = outer.dataPos + outer.size;
    return m;
}

// reads a matlab char matrix.
std::string readCharMatrix(const std::vector<unsigned char>& data, size_t pos) {
    MatMatrix m = readMatrixHeader(data, pos);
    if (m.cls != 4) return "";
    MatTag t = readTag(data, m.dataPos);
    std::string s;
    for (unsigned int i = 0; i < t.size; i++)
        s.push_back((char)data[t.dataPos + i]);
    return s;
}

// converts a matlab scalar tag to double.
double scalarFromDataTag(const std::vector<unsigned char>& data, const MatTag& t) {
    if (t.size == 0) return 0.0;

    switch (t.type) {
    case 1:
        return (signed char)data[t.dataPos];
    case 2:
        return data[t.dataPos];
    case 3:
        return (short)(data[t.dataPos] | (data[t.dataPos + 1] << 8));
    case 4:
        return (unsigned short)(data[t.dataPos] | (data[t.dataPos + 1] << 8));
    case 5:
        return readI32(data, t.dataPos);
    case 6:
        return readU32(data, t.dataPos);
    case 9: {
        double v;
        std::memcpy(&v, data.data() + t.dataPos, sizeof(double));
        return v;
    }
    default:
        return 0.0;
    }
}

// reads a scalar numeric matrix.
double readNumericMatrix(const std::vector<unsigned char>& data, size_t pos) {
    MatMatrix m = readMatrixHeader(data, pos);
    if (m.dataPos >= m.endPos) return 0.0;
    MatTag t = readTag(data, m.dataPos);
    return scalarFromDataTag(data, t);
}

// reads one frame of scene annotations.
std::vector<Detection> readStructArray(const std::vector<unsigned char>& data,
                                       size_t pos) {
    MatMatrix m = readMatrixHeader(data, pos);
    std::vector<Detection> result;
    if (m.cls != 2 || m.dims.empty()) return result;

    size_t cur = m.dataPos;
    MatTag lenTag = readTag(data, cur);
    int fieldLen = readI32(data, lenTag.dataPos);
    cur = lenTag.nextPos;

    MatTag namesTag = readTag(data, cur);
    std::vector<std::string> fieldNames;
    for (unsigned int start = 0; start < namesTag.size; start += fieldLen) {
        std::string name;
        for (int j = 0; j < fieldLen; j++) {
            char ch = (char)data[namesTag.dataPos + start + j];
            if (ch == 0) break;
            name.push_back(ch);
        }
        if (!name.empty())
            fieldNames.push_back(name);
    }
    cur = namesTag.nextPos;

    int numObj = 1;
    for (size_t i = 0; i < m.dims.size(); i++)
        numObj *= m.dims[i];

    for (int obj = 0; obj < numObj; obj++) {
        std::string category;
        int top = 0, bottom = 0, left = 0, right = 0;

        for (size_t f = 0; f < fieldNames.size(); f++) {
            MatTag fieldTag = readTag(data, cur);
            if (fieldTag.type == 14 && fieldTag.size > 0) {
                if (fieldNames[f] == "category")
                    category = readCharMatrix(data, cur);
                else if (fieldNames[f] == "top")
                    top = (int)std::round(readNumericMatrix(data, cur));
                else if (fieldNames[f] == "bottom")
                    bottom = (int)std::round(readNumericMatrix(data, cur));
                else if (fieldNames[f] == "left")
                    left = (int)std::round(readNumericMatrix(data, cur));
                else if (fieldNames[f] == "right")
                    right = (int)std::round(readNumericMatrix(data, cur));
            }
            cur = fieldTag.nextPos;
        }

        int cid = matClassIndex(category);
        if (cid >= 0) {
            Detection d;
            d.classId = cid;
            d.className = category;
            d.box = matClampRect(cv::Rect(left - 1, top - 1,
                                          right - left + 1,
                                          bottom - top + 1),
                                 TARGET_WIDTH, TARGET_HEIGHT);
            d.score = 1.0;
            if (d.box.area() > 0)
                result.push_back(d);
        }
    }

    return result;
}

} // namespace

// reads scene boxes from the mat file and saves them as csv.
std::vector<std::vector<Detection> > parseSceneGroundTruthMat(const std::string& matPath,
                                                              const std::string& csvPath) {
    std::vector<unsigned char> data = readFileBytes(matPath);
    std::vector<std::vector<Detection> > frames;
    if (data.empty()) {
        std::cerr << "Warning: Could not read scene GT MAT file." << std::endl;
        return frames;
    }

    try {
        MatTag compressed = readTag(data, 128);
        if (compressed.type != 15)
            throw std::runtime_error("Expected miCOMPRESSED block.");

        std::vector<unsigned char> dec =
            inflateBytes(data.data() + compressed.dataPos, compressed.size);
        MatMatrix root = readMatrixHeader(dec, 0);
        if (root.name != "bboxes")
            std::cerr << "Warning: Unexpected MAT variable: " << root.name << std::endl;

        int frameCount = root.dims.size() >= 2 ? root.dims[1] : 0;
        size_t cur = root.dataPos;
        for (int i = 0; i < frameCount; i++) {
            MatTag cell = readTag(dec, cur);
            if (cell.type == 14 && cell.size > 0)
                frames.push_back(readStructArray(dec, cur));
            else
                frames.push_back(std::vector<Detection>());
            cur = cell.nextPos;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: MAT parse failed: " << e.what() << std::endl;
    }

    std::ofstream csv(csvPath.c_str());
    csv << "frame,class,left,top,right,bottom\n";
    for (size_t f = 0; f < frames.size(); f++) {
        for (size_t i = 0; i < frames[f].size(); i++) {
            const Detection& d = frames[f][i];
            csv << (f + 1) << "," << d.className << ","
                << d.box.x << "," << d.box.y << ","
                << (d.box.x + d.box.width - 1) << ","
                << (d.box.y + d.box.height - 1) << "\n";
        }
    }

    std::cout << "Parsed scene GT frames: " << frames.size() << std::endl;
    return frames;
}
