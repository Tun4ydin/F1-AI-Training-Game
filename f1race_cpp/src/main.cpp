#include <SDL.h>
#include <nlohmann/json.hpp>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#include "imgui_stdlib.h"
#include "safe_algorithm.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
constexpr int W = 1600, H = 900;
constexpr double PI = 3.14159265358979323846;
constexpr double CAR_LENGTH_M = 5.6;
constexpr double CAR_WIDTH_M = 2.0;
constexpr double DEFAULT_CAMERA_ZOOM = 8.0;
constexpr double MIN_CAMERA_ZOOM = 2.0;
constexpr double MAX_CAMERA_ZOOM = 16.0;
std::pair<float,float> carSpriteDimensions(double cameraScale) {
    // Match Python's build_car_sprite(): the car has a world-space footprint
    // and follows the same camera scale as the circuit.  The small lower
    // bounds only keep the detailed sprite readable at very distant zooms.
    return {
        float(std::max(18.0,std::round(CAR_LENGTH_M*cameraScale))),
        float(std::max(8.0,std::round(CAR_WIDTH_M*cameraScale)))
    };
}

template <class T> T clampv(T value, T low, T high) {
    return std::max(low, std::min(high, value));
}

struct V2 {
    double x{}, y{};
    V2 operator+(V2 b) const { return {x + b.x, y + b.y}; }
    V2 operator-(V2 b) const { return {x - b.x, y - b.y}; }
    V2 operator-() const { return {-x,-y}; }
    V2 operator*(double s) const { return {x * s, y * s}; }
    V2 operator/(double s) const { return {x / s, y / s}; }
};
double dot(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
double length(V2 a) { return std::sqrt(dot(a, a)); }
V2 unit(V2 a) { const double n = length(a); return n > 1e-9 ? a / n : V2{1, 0}; }
V2 normal(V2 a) { return {-a.y, a.x}; }
double cross(V2 a,V2 b) { return a.x*b.y-a.y*b.x; }
double angleOf(V2 a) { return std::atan2(a.y, a.x); }
double angleDelta(double a, double b) {
    double d = std::fmod(a - b + PI, PI * 2.0);
    if (d < 0) d += PI * 2.0;
    return d - PI;
}
V2 catmull(V2 p0,V2 p1,V2 p2,V2 p3,double t){double t2=t*t,t3=t2*t;return (p1*2+(p2-p0)*t+(p0*2-p1*5+p2*4-p3)*t2+(-p0+p1*3-p2*3+p3)*t3)*.5;}
V2 catmullTangent(V2 p0,V2 p1,V2 p2,V2 p3,double t){double t2=t*t;V2 d=(p2-p0)+(p0*2-p1*5+p2*4-p3)*(2*t)+(-p0+p1*3-p2*3+p3)*(3*t2);return length(d)>1e-6?unit(d):unit(p2-p1);}

SDL_Color rgb(uint32_t value, uint8_t alpha = 255) {
    return SDL_Color{uint8_t(value >> 16), uint8_t(value >> 8), uint8_t(value), alpha};
}
void color(SDL_Renderer* r, SDL_Color c) { SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a); }
void fill(SDL_Renderer* r, SDL_Rect q, SDL_Color c) { color(r, c); SDL_RenderFillRect(r, &q); }
void outline(SDL_Renderer* r, SDL_Rect q, SDL_Color c) { color(r, c); SDL_RenderDrawRect(r, &q); }

// Compact 5x7 vector bitmap font. Lowercase is intentionally rendered as
// uppercase so the native UI stays sharp without a runtime font dependency.
std::array<uint8_t, 7> glyph(char raw) {
    char c = char(std::toupper(static_cast<unsigned char>(raw)));
    static const std::unordered_map<char, std::array<uint8_t, 7>> g = {
        {'A',{14,17,17,31,17,17,17}}, {'B',{30,17,17,30,17,17,30}},
        {'C',{14,17,16,16,16,17,14}}, {'D',{30,17,17,17,17,17,30}},
        {'E',{31,16,16,30,16,16,31}}, {'F',{31,16,16,30,16,16,16}},
        {'G',{14,17,16,23,17,17,15}}, {'H',{17,17,17,31,17,17,17}},
        {'I',{14,4,4,4,4,4,14}}, {'J',{7,2,2,2,18,18,12}},
        {'K',{17,18,20,24,20,18,17}}, {'L',{16,16,16,16,16,16,31}},
        {'M',{17,27,21,21,17,17,17}}, {'N',{17,25,21,19,17,17,17}},
        {'O',{14,17,17,17,17,17,14}}, {'P',{30,17,17,30,16,16,16}},
        {'Q',{14,17,17,17,21,18,13}}, {'R',{30,17,17,30,20,18,17}},
        {'S',{15,16,16,14,1,1,30}}, {'T',{31,4,4,4,4,4,4}},
        {'U',{17,17,17,17,17,17,14}}, {'V',{17,17,17,17,17,10,4}},
        {'W',{17,17,17,21,21,21,10}}, {'X',{17,17,10,4,10,17,17}},
        {'Y',{17,17,10,4,4,4,4}}, {'Z',{31,1,2,4,8,16,31}},
        {'0',{14,17,19,21,25,17,14}}, {'1',{4,12,4,4,4,4,14}},
        {'2',{14,17,1,2,4,8,31}}, {'3',{30,1,1,14,1,1,30}},
        {'4',{2,6,10,18,31,2,2}}, {'5',{31,16,16,30,1,1,30}},
        {'6',{14,16,16,30,17,17,14}}, {'7',{31,1,2,4,8,8,8}},
        {'8',{14,17,17,14,17,17,14}}, {'9',{14,17,17,15,1,1,14}},
        {'.',{0,0,0,0,0,6,6}}, {',',{0,0,0,0,0,6,4}},
        {':',{0,6,6,0,6,6,0}}, {';',{0,6,6,0,6,4,8}},
        {'-',{0,0,0,31,0,0,0}}, {'_',{0,0,0,0,0,0,31}},
        {'+',{0,4,4,31,4,4,0}}, {'=',{0,0,31,0,31,0,0}},
        {'/',{1,2,2,4,8,8,16}}, {'\\',{16,8,8,4,2,2,1}},
        {'(',{2,4,8,8,8,4,2}}, {')',{8,4,2,2,2,4,8}},
        {'[',{14,8,8,8,8,8,14}}, {']',{14,2,2,2,2,2,14}},
        {'<',{2,4,8,16,8,4,2}}, {'>',{8,4,2,1,2,4,8}},
        {'!',{4,4,4,4,4,0,4}}, {'?',{14,17,1,2,4,0,4}},
        {'#',{10,31,10,10,31,10,0}}, {'%',{17,2,4,8,17,0,0}},
        {'*',{0,21,14,31,14,21,0}}, {'"',{10,10,0,0,0,0,0}},
        {'\'',{4,4,0,0,0,0,0}}, {'|',{4,4,4,4,4,4,4}},
        {' ',{0,0,0,0,0,0,0}}
    };
    auto it = g.find(c);
    return it == g.end() ? g.at('?') : it->second;
}

void text(SDL_Renderer* r, int x, int y, std::string s, SDL_Color c,
          int scale = 2, int maxChars = 1000) {
    color(r, c);
    int origin = x, count = 0;
    for (char ch : s) {
        if (count++ >= maxChars) break;
        if (ch == '\n') { y += 9 * scale; x = origin; continue; }
        auto rows = glyph(ch);
        for (int row = 0; row < 7; ++row)
            for (int col = 0; col < 5; ++col)
                if (rows[row] & (1 << (4 - col))) {
                    SDL_Rect px{x + col * scale, y + row * scale, scale, scale};
                    SDL_RenderFillRect(r, &px);
                }
        x += 6 * scale;
    }
}
std::string readFile(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    return {std::istreambuf_iterator<char>(in), {}};
}
bool writeFile(const fs::path& p, const std::string& value) {
    std::error_code ec; fs::create_directories(p.parent_path(), ec);
    std::ofstream out(p, std::ios::binary); out << value; return bool(out);
}
fs::path localData() { return fs::path(F1RACE_CPP_ROOT) / "saved_data"; }
fs::path legacyData() { return fs::path(F1RACE_PYTHON_DATA_DIR); }
std::vector<fs::path> filesFor(const std::string& kind, const std::string& ext) {
    std::map<std::string, fs::path> unique;
    for (const auto& root : {legacyData(), localData()}) {
        const fs::path dir = root / kind;
        std::error_code ec;
        if (!fs::exists(dir, ec)) continue;
        for (const auto& e : fs::directory_iterator(dir, ec))
            if (e.is_regular_file() && e.path().extension() == ext)
                unique[e.path().filename().string()] = e.path();
    }
    std::vector<fs::path> out;
    for (auto& [_, p] : unique) out.push_back(p);
    return out;
}

struct Track {
    struct Projection { double distance{}; V2 point{}; size_t segment{}; double ratio{}; double metres{}; double lateral{}; };
    std::string name{"Starter Ring"};
    std::vector<V2> points;
    std::vector<double> widths,grassWidths;
    std::vector<V2> pitlanePoints;
    std::vector<double> pitlaneWidths,pitlaneGrassWidths;
    std::vector<V2> pitCenterline;
    std::vector<double> pitCenterWidths;
    std::set<int> kerbs;
    json features = json::object();
    std::vector<double> cumulative;
    std::unordered_map<int64_t,std::vector<int>> spatial;
    double lengthM{};
    double geometryLength{};
    double startGeometryOffset{};

    void adjustAllWidths(double delta){for(size_t i=0;i<widths.size();++i){widths[i]=clampv(widths[i]+delta,6.0,44.0);if(i<grassWidths.size())grassWidths[i]=std::max(grassWidths[i],widths[i]+4.0);}}
    void adjustAllGrassWidths(double delta){for(size_t i=0;i<grassWidths.size();++i){double minGrass=(i<widths.size()?widths[i]+4.0:16.0);grassWidths[i]=clampv(grassWidths[i]+delta,minGrass,120.0);}}
    void setAllWidths(double roadWidth){roadWidth=clampv(roadWidth,6.0,44.0);for(size_t i=0;i<widths.size();++i){widths[i]=roadWidth;if(i<grassWidths.size())grassWidths[i]=std::max(grassWidths[i],roadWidth+4.0);}}
    void setAllGrassWidths(double grassWidth){for(size_t i=0;i<grassWidths.size();++i){double minGrass=(i<widths.size()?widths[i]+4.0:16.0);grassWidths[i]=clampv(grassWidth,minGrass,120.0);}}
    void adjustAllPitWidths(double delta){for(size_t i=0;i<pitlaneWidths.size();++i){pitlaneWidths[i]=clampv(pitlaneWidths[i]+delta,4.0,18.0);if(i<pitlaneGrassWidths.size())pitlaneGrassWidths[i]=std::max(pitlaneGrassWidths[i],pitlaneWidths[i]+2.0);}}
    void adjustAllPitGrassWidths(double delta){for(size_t i=0;i<pitlaneGrassWidths.size();++i){double minGrass=(i<pitlaneWidths.size()?pitlaneWidths[i]+2.0:8.0);pitlaneGrassWidths[i]=clampv(pitlaneGrassWidths[i]+delta,minGrass,60.0);}}

    bool load(const fs::path& path) {
        try {
            auto d = json::parse(readFile(path));
            name = d.value("name", path.stem().string());
            points.clear(); widths.clear(); grassWidths.clear();pitlanePoints.clear();pitlaneWidths.clear();pitlaneGrassWidths.clear(); kerbs.clear();
            for (const auto& p : d.at("points")) points.push_back({p[0], p[1]});
            const double defaultWidth = d.value("road_width_m", 9.0);
            if (d.contains("road_widths_m"))
                for (double w : d["road_widths_m"]) widths.push_back(w);
            widths.resize(points.size(), defaultWidth);
            if(d.contains("grass_widths_m"))for(double w:d["grass_widths_m"])grassWidths.push_back(w);
            grassWidths.resize(points.size(),std::max(defaultWidth+8.0,20.0));
            if(d.contains("pitlane_points"))for(const auto&p:d["pitlane_points"])pitlanePoints.push_back({p[0],p[1]});
            if(d.contains("pitlane_road_widths_m"))for(double w:d["pitlane_road_widths_m"])pitlaneWidths.push_back(w);
            if(d.contains("pitlane_grass_widths_m"))for(double w:d["pitlane_grass_widths_m"])pitlaneGrassWidths.push_back(w);
            pitlaneWidths.resize(pitlanePoints.size(),6.0);pitlaneGrassWidths.resize(pitlanePoints.size(),16.0);
            std::vector<int> rawKerbs;if (d.contains("kerb_points")) for (int k : d["kerb_points"])rawKerbs.push_back(k);
            features = d.value("features", json::object());
            int featureScale=1;if(d.value("geometry",std::string("spline"))!="sampled"&&points.size()>=4){
                const int samples=clampv(int(std::ceil(240.0/points.size())),4,14);
                featureScale=samples;
                auto rawPoints=points;
                auto rawWidths=widths;
                auto rawGrass=grassWidths;
                std::set<int> autoKerbs;
                if(rawKerbs.empty()){
                    for(size_t i=0;i<rawPoints.size();++i){
                        V2 pt=rawPoints[i], inP=rawPoints[(i+rawPoints.size()-1)%rawPoints.size()]-pt, outP=rawPoints[(i+1)%rawPoints.size()]-pt;
                        if(length(inP)>1e-6&&length(outP)>1e-6){
                            double dt=clampv(dot(unit(inP),unit(outP)),-1.0,1.0);
                            double deg=180.0-std::acos(dt)*180.0/PI;
                            if(deg>=25.0) autoKerbs.insert(int(i));
                        }
                    }
                } else {
                    for(int k:rawKerbs) autoKerbs.insert(k);
                }
                points.clear();widths.clear();grassWidths.clear();
                for(size_t i=0;i<rawPoints.size();++i){
                    for(int sample=0;sample<samples;++sample){
                        double t=double(sample)/samples;
                        size_t n=rawPoints.size();
                        bool nearCorner = (autoKerbs.count(int(i)) && t < 0.35) ||
                                          (autoKerbs.count(int((i+1)%n)) && t > 0.65) ||
                                          (autoKerbs.count(int(i)) && autoKerbs.count(int((i+1)%n)));
                        if(nearCorner) kerbs.insert(int(points.size()));
                        points.push_back(catmull(rawPoints[(i+n-1)%n],rawPoints[i],rawPoints[(i+1)%n],rawPoints[(i+2)%n],t));
                        widths.push_back(rawWidths[i]*(1-t)+rawWidths[(i+1)%n]*t);
                        grassWidths.push_back(rawGrass[i]*(1-t)+rawGrass[(i+1)%n]*t);
                    }
                }
            } else {
                if(rawKerbs.empty()){
                    for(size_t i=0;i<points.size();++i){
                        V2 pt=points[i], inP=points[(i+points.size()-1)%points.size()]-pt, outP=points[(i+1)%points.size()]-pt;
                        if(length(inP)>1e-6&&length(outP)>1e-6){
                            double dt=clampv(dot(unit(inP),unit(outP)),-1.0,1.0);
                            double deg=180.0-std::acos(dt)*180.0/PI;
                            if(deg>=6.0) kerbs.insert(int(i));
                        }
                    }
                } else {
                    for(int k:rawKerbs) kerbs.insert(k);
                }
            }
            if(featureScale>1){for(const char* key:{"start_finish","pit_entry","pit_exit","drs_detection","drs_entry","drs_exit"})if(features.contains(key)&&features[key].is_number_integer())features[key]=features[key].get<int>()*featureScale;if(features.contains("sectors")&&features["sectors"].is_array())for(auto&v:features["sectors"])if(v.is_number_integer())v=v.get<int>()*featureScale;}
            if(!features.contains("start_finish"))features["start_finish"]=0;if(!features.contains("sectors"))features["sectors"]=json::array();if(!features.contains("pit_entry"))features["pit_entry"]=nullptr;if(!features.contains("pit_exit"))features["pit_exit"]=nullptr;if(!features.contains("pit_boxes"))features["pit_boxes"]=json::array();if(!features.contains("pit_start_finish"))features["pit_start_finish"]=nullptr;
            rebuild();
            if (d.contains("declared_length_m") && d["declared_length_m"].is_number())
                lengthM = d["declared_length_m"];
            return points.size() >= 3;
        } catch (...) { return false; }
    }
    void rebuild() {
        cumulative.assign(points.size() + 1, 0.0);
        for (size_t i = 0; i < points.size(); ++i)
            cumulative[i + 1] = cumulative[i] + length(points[(i + 1) % points.size()] - points[i]);
        geometryLength = cumulative.empty() ? 0.0 : cumulative.back();
        lengthM = geometryLength;
        widths.resize(points.size(), 9.0);
        grassWidths.resize(points.size(),20.0);
        startGeometryOffset=0;if(features.contains("start_finish")&&features["start_finish"].is_number_integer()&&!points.empty()){int start=clampv(features["start_finish"].get<int>(),0,int(points.size())-1);startGeometryOffset=cumulative[size_t(start)];}
        spatial.clear();constexpr double cell=40.0;auto key=[](int x,int y){return int64_t((uint64_t(uint32_t(x))<<32)|uint32_t(y));};for(size_t i=0;i<points.size();++i){V2 a=points[i],b=points[(i+1)%points.size()];double margin=std::max(grassWidths[i],grassWidths[(i+1)%points.size()])*.5+3;int left=int(std::floor((std::min(a.x,b.x)-margin)/cell)),right=int(std::floor((std::max(a.x,b.x)+margin)/cell)),top=int(std::floor((std::min(a.y,b.y)-margin)/cell)),bottom=int(std::floor((std::max(a.y,b.y)+margin)/cell));for(int x=left;x<=right;++x)for(int y=top;y<=bottom;++y)spatial[key(x,y)].push_back(int(i));}
        rebuildPitlane();
    }
    void rebuildPitlane(){pitCenterline.clear();pitCenterWidths.clear();if(pitlanePoints.empty()||points.empty()||!features.contains("pit_entry")||!features.contains("pit_exit")||!features["pit_entry"].is_number_integer()||!features["pit_exit"].is_number_integer())return;size_t entry=size_t(clampv(features["pit_entry"].get<int>(),0,int(points.size())-1)),exit=size_t(clampv(features["pit_exit"].get<int>(),0,int(points.size())-1));auto tangentAt=[&](size_t i){return unit(points[(i+1)%points.size()]-points[(i+points.size()-1)%points.size()]);};auto edge=[&](size_t i,V2 toward){V2 tangent=tangentAt(i),n=normal(tangent);double side=dot(toward-points[i],n)>=0?1:-1;return points[i]+n*(widths[i]*.5*side);};auto curve=[](V2 start,V2 end,V2 startTangent,V2 endTangent){std::vector<V2>out;double distance=length(end-start);if(distance<=1e-6){out.push_back(start);return out;}double handle=clampv(distance*.34,3.0,22.0);V2 c1=start+unit(startTangent)*handle,c2=end-unit(endTangent)*handle;int segments=clampv(int(std::ceil(distance/3.5)),7,18);for(int i=0;i<=segments;++i){double t=double(i)/segments,u=1-t;out.push_back(start*(u*u*u)+c1*(3*u*u*t)+c2*(3*u*t*t)+end*(t*t*t));}return out;};V2 entryDirection=pitlanePoints.size()>1?pitlanePoints[1]-pitlanePoints[0]:edge(exit,pitlanePoints.back())-pitlanePoints[0],exitDirection=pitlanePoints.size()>1?pitlanePoints.back()-pitlanePoints[pitlanePoints.size()-2]:entryDirection;auto incoming=curve(edge(entry,pitlanePoints.front()),pitlanePoints.front(),tangentAt(entry),entryDirection),outgoing=curve(pitlanePoints.back(),edge(exit,pitlanePoints.back()),exitDirection,tangentAt(exit));pitCenterline=incoming;if(!pitCenterline.empty())pitCenterline.pop_back();pitCenterline.insert(pitCenterline.end(),pitlanePoints.begin(),pitlanePoints.end());if(!outgoing.empty())pitCenterline.insert(pitCenterline.end(),outgoing.begin()+1,outgoing.end());for(V2 point:pitCenterline){double best=1e9,value=6;for(size_t i=0;i+1<pitlanePoints.size();++i){V2 d=pitlanePoints[i+1]-pitlanePoints[i];double t=dot(d,d)>1e-9?clampv(dot(point-pitlanePoints[i],d)/dot(d,d),0.0,1.0):0,dist=length(point-(pitlanePoints[i]+d*t));if(dist<best){best=dist;double a=i<pitlaneWidths.size()?pitlaneWidths[i]:6,b=i+1<pitlaneWidths.size()?pitlaneWidths[i+1]:a;value=a*(1-t)+b*t;}}pitCenterWidths.push_back(value);}}
    std::pair<V2,V2> at(double metres) const {
        if (points.empty()) return {{}, {1,0}};
        double s = std::fmod(metres, std::max(1.0, lengthM)); if (s < 0) s += lengthM;
        s = std::fmod(s*geometryLength/std::max(1.0,lengthM)+startGeometryOffset,std::max(1.0,geometryLength));
        auto it = std::upper_bound(cumulative.begin(), cumulative.end(), s);
        size_t i = clampv<size_t>(size_t(std::distance(cumulative.begin(), it) - 1), 0, points.size()-1);
        V2 a = points[i], b = points[(i+1)%points.size()];
        double seg = std::max(1e-9, cumulative[i+1]-cumulative[i]);
        return {a + (b-a)*((s-cumulative[i])/seg), unit(b-a)};
    }
    double widthAt(double metres) const {
        if (points.empty()) return 9.0;
        double s = std::fmod(metres, std::max(1.0,lengthM)); if(s<0)s+=lengthM;
        s = std::fmod(s*geometryLength/std::max(1.0,lengthM)+startGeometryOffset,std::max(1.0,geometryLength));
        auto it=std::upper_bound(cumulative.begin(),cumulative.end(),s);
        size_t i=clampv<size_t>(size_t(std::distance(cumulative.begin(),it)-1),0,points.size()-1);
        double t=(s-cumulative[i])/std::max(1e-9,cumulative[i+1]-cumulative[i]);
        return widths[i]*(1-t)+widths[(i+1)%points.size()]*t;
    }
    double curvature(double metres, double ahead = 25.0) const {
        auto [_, a] = at(metres); auto [__, b] = at(metres + ahead);
        return clampv(angleDelta(angleOf(b), angleOf(a)) / 1.2, -1.0, 1.0);
    }
    Projection project(V2 point,bool fallback=true) const {
        Projection best{std::numeric_limits<double>::infinity(),{},0,0,0,0};
        if(points.empty())return best;
        std::vector<int> candidates;constexpr double cell=40.0;auto key=[](int x,int y){return int64_t((uint64_t(uint32_t(x))<<32)|uint32_t(y));};int cx=int(std::floor(point.x/cell)),cy=int(std::floor(point.y/cell));for(int x=cx-1;x<=cx+1;++x)for(int y=cy-1;y<=cy+1;++y){auto found=spatial.find(key(x,y));if(found!=spatial.end())candidates.insert(candidates.end(),found->second.begin(),found->second.end());}if(candidates.empty()&&fallback){candidates.resize(points.size());for(size_t i=0;i<points.size();++i)candidates[i]=int(i);}for(int raw:candidates){size_t i=size_t(raw);V2 a=points[i],delta=points[(i+1)%points.size()]-a;double denominator=dot(delta,delta);double ratio=denominator>1e-12?clampv(dot(point-a,delta)/denominator,0.0,1.0):0.0;V2 nearest=a+delta*ratio;double distance=length(point-nearest);if(distance<best.distance){double geometry=cumulative[i]+length(delta)*ratio;double relative=std::fmod(geometry-startGeometryOffset+geometryLength,std::max(1.0,geometryLength));V2 tangent=unit(delta);best={distance,nearest,i,ratio,relative*lengthM/std::max(1.0,geometryLength),dot(point-nearest,normal(tangent))*lengthM/std::max(1.0,geometryLength)};}}
        return best;
    }
    double progressMetres(V2 point) const { return project(point).metres; }
    double widthAtProjection(const Projection&p)const{return widths.empty()?9.0:widths[p.segment]*(1-p.ratio)+widths[(p.segment+1)%widths.size()]*p.ratio;}
    struct PitNearest { double distance{1e18}; V2 point{}; int segment{-1}; double ratio{}; };
    PitNearest pitlaneNearest(V2 point) const {
        PitNearest best;
        if(pitCenterline.size()<2) return best;
        for(size_t i=0;i+1<pitCenterline.size();++i){
            V2 a=pitCenterline[i],d=pitCenterline[i+1]-a;
            double t=dot(d,d)>1e-12?clampv(dot(point-a,d)/dot(d,d),0.0,1.0):0;
            V2 nearest=a+d*t;
            double distance=length(point-nearest);
            if(distance<best.distance){best={distance,nearest,int(i),t};}
        }
        return best;
    }
    V2 pitlanePointAhead(V2 point, double lookahead=14.0) const {
        if(pitCenterline.size()<2) return {};
        auto nearest=pitlaneNearest(point);
        if(nearest.segment<0) return {};
        V2 target=nearest.point;
        double remaining=std::max(0.0,lookahead);
        for(size_t i=size_t(nearest.segment);i+1<pitCenterline.size();++i){
            V2 endpoint=pitCenterline[i+1];
            double segLen=length(endpoint-target);
            if(segLen>=remaining && segLen>1e-9){
                double ratio=remaining/segLen;
                return target+(endpoint-target)*ratio;
            }
            remaining-=segLen;
            target=endpoint;
        }
        return pitCenterline.back();
    }
    bool inPitlane(V2 point) const {
        if(pitCenterline.size()<2)return false;const double scale=lengthM/std::max(1.0,geometryLength);double best=std::numeric_limits<double>::infinity(),width=6;
        for(size_t i=0;i+1<pitCenterline.size();++i){V2 a=pitCenterline[i],d=pitCenterline[i+1]-a;double t=dot(d,d)>1e-12?clampv(dot(point-a,d)/dot(d,d),0.0,1.0):0;double distance=length(point-(a+d*t))*scale;if(distance<best){best=distance;double wa=i<pitCenterWidths.size()?pitCenterWidths[i]:6,wb=i+1<pitCenterWidths.size()?pitCenterWidths[i+1]:wa;width=wa*(1-t)+wb*t;}}
        return best<=width*.5;
    }
    std::string surface(V2 point)const{
        if(inPitlane(point))return "pitlane";auto p=project(point,false);if(!std::isfinite(p.distance))return "wall";double localWidth=widthAtProjection(p);if(p.distance*lengthM/std::max(1.0,geometryLength)<=localWidth*.5)return "asphalt";if(kerbs.count(int(p.segment))&&p.distance*lengthM/std::max(1.0,geometryLength)<=(localWidth+2)*.5)return "kerb";double ga=grassWidths.empty()?localWidth+10:grassWidths[p.segment],gb=grassWidths.empty()?ga:grassWidths[(p.segment+1)%grassWidths.size()];double grass=ga*(1-p.ratio)+gb*p.ratio;return p.distance*lengthM/std::max(1.0,geometryLength)<=grass*.5?"grass":"wall";
    }
    double featureMetres(const char*key)const{if(!features.contains(key)||!features[key].is_number_integer()||points.empty())return -1;size_t i=size_t(clampv(features[key].get<int>(),0,int(points.size())-1));double relative=std::fmod(cumulative[i]-startGeometryOffset+geometryLength,std::max(1.0,geometryLength));return relative*lengthM/std::max(1.0,geometryLength);}
    bool crossedTimingLine(V2 previous,V2 current,bool pit=false)const{V2 centre,tangent;double half=4.5;if(pit){if(!features.contains("pit_start_finish")||!features["pit_start_finish"].is_number_integer()||pitlanePoints.empty())return false;size_t i=size_t(clampv(features["pit_start_finish"].get<int>(),0,int(pitlanePoints.size())-1)),before=i?i-1:i,after=std::min(i+1,pitlanePoints.size()-1);centre=pitlanePoints[i];tangent=unit(pitlanePoints[after]-pitlanePoints[before]);half=(i<pitlaneWidths.size()?pitlaneWidths[i]:6)*.5;}else{if(points.empty())return false;size_t i=features.contains("start_finish")&&features["start_finish"].is_number_integer()?size_t(clampv(features["start_finish"].get<int>(),0,int(points.size())-1)):0;centre=points[i];tangent=unit(points[(i+1)%points.size()]-points[(i+points.size()-1)%points.size()]);half=widths[i]*.5;}double before=dot(previous-centre,tangent),after=dot(current-centre,tangent);if(before>=-1e-4||after<0||after-before<=1e-9)return false;V2 crossing=previous+(current-previous)*(-before/(after-before));return std::abs(dot(crossing-centre,normal(tangent)))<=half+CAR_WIDTH_M*.5;}
    std::vector<V2> pitBoxes()const{std::vector<V2>out;if(!features.contains("pit_boxes")||!features["pit_boxes"].is_array())return out;for(const auto&value:features["pit_boxes"])if(value.is_number_integer()){int i=value.get<int>();if(!pitlanePoints.empty()&&i>=0&&i<int(pitlanePoints.size()))out.push_back(pitlanePoints[size_t(i)]);else if(!points.empty())out.push_back(points[size_t((i%int(points.size())+int(points.size()))%int(points.size()))]);}return out;}
    json toJson() const {
        json d; d["name"]=name; d["geometry"]="sampled"; d["declared_length_m"]=lengthM;
        d["road_width_m"]=widths.empty()?9.0:widths.front(); d["features"]=features;
        d["points"]=json::array(); for(auto p:points)d["points"].push_back({p.x,p.y});
        d["road_widths_m"]=widths; d["kerb_points"]=kerbs;
        d["pitlane_points"]=json::array();for(auto p:pitlanePoints)d["pitlane_points"].push_back({p.x,p.y});d["grass_widths_m"]=grassWidths;
        d["pitlane_road_widths_m"]=pitlaneWidths; d["pitlane_grass_widths_m"]=pitlaneGrassWidths;
        return d;
    }
};

struct Transform {
    double scale{1}, ox{}, oy{};
    SDL_FPoint screen(V2 p) const { return {float(ox+p.x*scale),float(oy+p.y*scale)}; }
    V2 world(int x,int y) const { return {(x-ox)/scale,(y-oy)/scale}; }
};
Transform fitTrack(const Track& t, SDL_Rect area, double padding=35) {
    if(t.points.empty()) return {1,double(area.x+area.w/2),double(area.y+area.h/2)};
    double minx=t.points[0].x,maxx=minx,miny=t.points[0].y,maxy=miny;
    for(auto p:t.points){minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);}for(auto p:t.pitlanePoints){minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);}
    double sx=(area.w-padding*2)/std::max(1.0,maxx-minx), sy=(area.h-padding*2)/std::max(1.0,maxy-miny);
    double s=std::min(sx,sy);
    return {s,area.x+(area.w-(maxx-minx)*s)/2-minx*s,area.y+(area.h-(maxy-miny)*s)/2-miny*s};
}

void ribbonSegment(SDL_Renderer* r, SDL_FPoint af, SDL_FPoint bf,
                   float startWidth, float endWidth, SDL_Color c) {
    V2 a{af.x, af.y}, b{bf.x, bf.y};
    V2 n = normal(unit(b - a));
    V2 p0 = a + n * (startWidth * .5), p1 = a - n * (startWidth * .5);
    V2 p2 = b - n * (endWidth * .5), p3 = b + n * (endWidth * .5);
    SDL_Vertex vertices[4] = {
        {{float(p0.x),float(p0.y)},c,{0,0}}, {{float(p1.x),float(p1.y)},c,{0,0}},
        {{float(p2.x),float(p2.y)},c,{0,0}}, {{float(p3.x),float(p3.y)},c,{0,0}}
    };
    const int indices[6] = {0,1,2,0,2,3};
    SDL_RenderGeometry(r,nullptr,vertices,4,indices,6);
}

void roadJoint(SDL_Renderer* r, SDL_FPoint center, float radius, SDL_Color c) {
    constexpr int steps = 18;
    std::array<SDL_Vertex, steps + 1> vertices{};
    std::array<int, steps * 3> indices{};
    vertices[0] = {center,c,{0,0}};
    for(int i=0;i<steps;++i) {
        double a = i * PI * 2.0 / steps;
        vertices[i+1]={{center.x+float(std::cos(a)*radius),center.y+float(std::sin(a)*radius)},c,{0,0}};
        indices[i*3]=0;indices[i*3+1]=i+1;indices[i*3+2]=(i+1)%steps+1;
    }
    SDL_RenderGeometry(r,nullptr,vertices.data(),int(vertices.size()),indices.data(),int(indices.size()));
}

struct Controls { double steer{}, throttle{}, brake{}, overtake{}, recharge{}, pitRequest{}; int pitTyre{1}; };
struct Brain {
    std::string name{"EMPTY BRAIN"};
    std::map<std::string,double> p;
    std::map<std::string,double> config{{"steering",1.0},{"aggression",.82},{"braking",.55},{"recovery",.8},{"mutation",.22}};
    std::vector<std::vector<double>> weights;
    std::string source;
    fai::Program program;
    Brain(){resetWeights();}
    void resetWeights(){double turn=config["steering"],recovery=config["recovery"];weights={{-.40*turn,-1.0*turn,0,1.0*turn,.40*turn,.75*recovery,0,0,0},{0,0,1,0,0,0,-1,-.4,0}};}
    static Brain load(const fs::path& path) {
        Brain b;
        try { auto d=json::parse(readFile(path)); b.name=d.value("name",path.stem().string()); b.source=d.value("source","");if(d.contains("algorithm")&&d["algorithm"].is_object())for(auto&[k,v]:d["algorithm"].items())if(v.is_number())b.config[k]=v;if(d.contains("weights")&&d["weights"].is_array()){b.weights.clear();for(const auto&row:d["weights"])if(row.is_array()){std::vector<double>values;for(const auto&value:row)if(value.is_number())values.push_back(value);b.weights.push_back(std::move(values));}}else if(b.source.empty())b.resetWeights();
            if(d.contains("parameters")&&d["parameters"].is_object()) for(auto&[k,v]:d["parameters"].items()) if(v.is_number())b.p[k]=v;
            if(!b.source.empty())b.program.compile(b.source);
        } catch(...){} return b;
    }
    void setSource(const std::string& value){source=value;program.compile(value);for(const auto&[key,spec]:program.parameters())if(!p.count(key))p[key]=spec.defaultValue;}
    double get(const std::string& key,double fallback)const{auto i=p.find(key);return i==p.end()?fallback:i->second;}
    Controls think(const std::unordered_map<std::string,double>& inputs)const{
        if(program.valid()&&!source.empty()){auto out=program.run(inputs,p);return{out.steering,out.throttle,out.brake,out.overtake,out.recharge,out.pitRequest,out.pitTyre};}
        auto in=[&](const char*key){auto i=inputs.find(key);return i==inputs.end()?0.0:i->second;};std::array<double,8>state={in("far_left"),in("left"),in("forward"),in("right"),in("far_right"),in("heading_error"),in("speed"),in("dirty_tyres")};std::array<double,2>output{};for(size_t row=0;row<2&&row<weights.size();++row){double sum=weights[row].empty()?0:weights[row].back();for(size_t i=0;i<state.size()&&i+1<weights[row].size();++i)sum+=state[i]*weights[row][i];output[row]=std::tanh(sum);}double open=in("right")-in("left")+(in("far_right")-in("far_left"))*.4;double steer=clampv(output[0],-1.0,1.0);double target=in("forward")*(.9+config.at("aggression")*.25)-std::abs(steer)*.25;double throttle=clampv((target-in("speed"))*(.4+config.at("braking")*.5),0.0,1.0)*(1-in("dirty_tyres")*.5),brake=0;if(in("forward")<.25){if(std::abs(open)>.01)steer=open>0?.9:-.9;throttle=.05;brake=in("is_hybrid")>.5?.35:0;}bool pit=in("pit_available")>.5&&(in("tyre_wear")>=.65||in("puncture")>.5);bool deploy=in("is_hybrid")>.5&&in("battery")>=.18&&in("forward")>=.72&&std::abs(steer)<=.2&&brake<=.05;bool recharge=in("is_hybrid")>.5&&in("battery")<=.2;return{steer,throttle,brake,double(deploy),double(recharge),double(pit),in("rain")>=.45?3:1};
    }
    Brain mutate(std::mt19937& rng,double amount=.12)const{
        Brain b=*this; std::normal_distribution<double> n(0,amount); std::uniform_real_distribution<double> u(0,1);
        static const std::vector<std::pair<std::string,double>> defaults={{"waypoint_gain",.95},{"heading_gain",.65},{"center_gain",.5},{"corner_caution",.48},{"throttle_gain",2.1},{"brake_gain",1.3},{"steering_smoothing",.15}};
        if(b.program.valid()&&!b.program.parameters().empty()){for(const auto&[key,spec]:b.program.parameters()){double current=b.get(key,spec.defaultValue);if(u(rng)<.55)current+=n(rng)*(spec.high-spec.low);b.p[key]=clampv(current,spec.low,spec.high);}}
        else{if(!b.weights.empty()){for(auto&row:b.weights)for(double&value:row)if(u(rng)<.35)value+=n(rng);}else{for(auto&[k,v]:defaults)if(!b.p.count(k))b.p[k]=v;for(auto&[_,v]:b.p)if(u(rng)<.55)v+=n(rng)*std::max(.15,std::abs(v));}}
        return b;
    }
};

struct Car {
    std::string name{"NOVA"}; SDL_Color col{50,220,190,255}; Brain brain; bool hybrid{};
    double s{},lateral{},speed{},angle{},battery{100},fuel{50},wear{},throttle{},brake{},steer{},health{100},dirty{};
    double rpm{4000},fitness{},bestLap{1e9},lapStart{},lastLap{},raceDistance{},forwardDistance{},reverseDistance{},offTrackFrames{},controlPenalty{},collisionPenalty{},lowSpeedSeconds{},finishTime{-1};
    double understeer{},oversteer{},traction{1},tireSlip{},angularVelocity{},slipstream{},batteryRegen{},drsGap{1e9},pitTimer{};
    double carAhead{},carAheadDistance{1},carAheadSide{},closingSpeed{},passingSide{},raceAggression{},aggressionError{},gapLeader{},gapNext{};
    double apexDistance{1},apexCurvature{},longCurvature70{},longCurvature110{},longCurvature160{};
    double stagnantFrames{},aggressionMistakeFrames{},aggressionMistakeCooldown{};
    double pitExitStraightFrames{},pitExitGuidanceFrames{};
    int gear{1},lap{},checkpoint{},overtakes{},tyreCompound{},tyreLaps{},pitstops{},requestedTyre{1},startingPosition{1},racePosition{1},fieldSize{1},sensorFrame{},trackLimits{};
    bool deploying{},regen{},drsEligible{},drsActive{},drsInZone{},outsideLimits{},carCollision{},passing{},pitRequested{},inPitlane{},puncture{},alive{true},finished{},removed{};
    bool pitEntryCommitted{},redFlagPitStopped{};
    V2 position{},velocity{};bool poseReady{};double previousProgress{-1};std::array<double,12> opponentData{};std::array<double,3> opponentPresence{};std::array<double,9> rayCache{};bool raysReady{};std::set<int> overtakeCandidates;std::map<int,int> overtakeCooldowns;

    static bool crossed(double previous,double current,double marker,double trackLength){if(marker<0||trackLength<=0)return false;if(current>=previous)return previous<marker&&marker<=current;return marker>previous||marker<=current;}
    void advanceAfterFinish(const Track&track,double dt){if(removed)return;double frameDt=clampv(dt*60.0,0.0,3.0),current=length(velocity),remaining=std::max(0.0,current-.1*.028*frameDt);if(remaining<=1e-6){velocity={};speed=0;removed=true;return;}double progress=track.progressMetres(position);V2 target=track.at(progress+clampv(remaining*28,12.0,26.0)).first,desired=target-position,direction=current>1e-9?unit(velocity):V2{std::cos(angle),std::sin(angle)};if(length(desired)>1e-9)direction=unit(direction+(unit(desired)-direction)*clampv(.10*frameDt,0.0,.35));velocity=direction*remaining;angle=angleOf(direction);brake=.1;throttle=steer=0;position=position+velocity*frameDt;auto projection=track.project(position);s=projection.metres;lateral=projection.lateral;speed=remaining*216;updateDrivetrain(0);}
    void updateDrivetrain(double throttleValue){static constexpr std::array<double,8> ratios={3.20,2.55,2.08,1.72,1.45,1.25,1.16,.94};V2 heading{std::cos(angle),std::sin(angle)};double forwardSpeed=std::max(0.0,dot(velocity,heading)),ratio=clampv(forwardSpeed/1.67,0.0,1.0);gear=8;for(int i=0;i<8;++i)if(ratio<=std::min(1.0,.94/ratios[size_t(i)])){gear=i+1;break;}double low=gear==1?0:std::min(1.0,.94/ratios[size_t(gear-2)]),high=std::min(1.0,.94/ratios[size_t(gear-1)]),within=clampv((ratio-low)/std::max(1e-9,high-low),0.0,1.0);rpm=forwardSpeed<.025?4000+clampv(throttleValue,0.0,1.0)*2500:4000+within*9000;rpm=clampv(rpm,4000.0,13000.0);}
    double drivetrainPower()const{static constexpr std::array<double,8>ratios={3.20,2.55,2.08,1.72,1.45,1.25,1.16,.94};double strength=ratios[size_t(gear-1)]/ratios[0],rpmFraction=clampv((rpm-4000)/9000,0.0,1.0),redline=clampv((rpmFraction-.67)/.33,0.0,1.0);return(.10+strength*.90)*(1-.20*redline*redline);}
    std::array<double,9> raycasts(const Track&track)const{std::array<double,9>values{};static constexpr std::array<double,9>angles={-90,-70,-35,-18,0,18,35,70,90};for(size_t i=0;i<angles.size();++i){double a=angle+angles[i]*PI/180.0;V2 direction{std::cos(a),std::sin(a)};double previous=0,distance=4;for(;distance<=120;distance+=4){std::string surface=track.surface(position+direction*distance);if(surface!="asphalt"&&surface!="kerb"&&surface!="pitlane"){double lo=previous,hi=distance;for(int refine=0;refine<2;++refine){double mid=(lo+hi)*.5;std::string probe=track.surface(position+direction*mid);if(probe=="asphalt"||probe=="kerb"||probe=="pitlane")lo=mid;else hi=mid;}distance=lo;break;}previous=distance;}values[i]=std::min(distance,120.0)/120.0;}return values;}
    // Scans the curvature profile ahead for the sharpest bend and reports how far away it is (0..1 over a 180m horizon, 1 = none found) and its signed severity (-1..1). Purely informational: brains may ignore it, aim for it, or build their own apex logic from the raw corner_curvature_* samples instead.
    std::pair<double,double> findApex(const Track&track,double metres)const{double bestDistance=1.0,bestCurvature=0,bestMagnitude=0;constexpr double horizon=180.0,step=10.0,window=14.0;for(double d=step;d<=horizon;d+=step){double c=track.curvature(metres+d-window*.5,window);if(std::abs(c)>bestMagnitude){bestMagnitude=std::abs(c);bestDistance=d/horizon;bestCurvature=c;}}if(bestMagnitude<.12)return{1.0,0.0};return{bestDistance,bestCurvature};}
    std::unordered_map<std::string,double> controllerInputs(const Track&track,double rain){
        auto projection=track.project(position);V2 tangent=unit(track.points[(projection.segment+1)%track.points.size()]-track.points[projection.segment]),heading{std::cos(angle),std::sin(angle)},localNormal=normal(heading);if(!raysReady||sensorFrame%3==0){rayCache=raycasts(track);auto apex=findApex(track,projection.metres);apexDistance=apex.first;apexCurvature=apex.second;longCurvature70=track.curvature(projection.metres,70);longCurvature110=track.curvature(projection.metres,110);longCurvature160=track.curvature(projection.metres,160);raysReady=true;}++sensorFrame;double racingOffset=clampv(projection.lateral/std::max(1.0,track.widthAtProjection(projection)*.5),-1.0,1.0),headingError=std::atan2(cross(heading,tangent),dot(heading,tangent))/PI;
        std::unordered_map<std::string,double>v;auto put=[&](const char*n,double value){v[n]=value;};put("far_left",rayCache[1]);put("left",rayCache[2]);put("forward",rayCache[4]);put("right",rayCache[6]);put("far_right",rayCache[7]);put("heading_error",headingError);put("speed",clampv(length(velocity)/1.7,0.0,1.0));put("dirty_tyres",dirty/180);put("tyre_wear",wear);put("tyre_age",tyreLaps);put("fuel",clampv(fuel/110,0.0,1.0));put("fuel_kg",fuel);put("health",health/100);put("puncture",wear>=1.0);put("rain",rain);put("slipstream",slipstream);put("lap",lap);put("lap_progress",projection.metres/std::max(1.0,track.lengthM));put("pitstops",pitstops);put("pit_available",track.pitBoxes().empty()?0:1);for(int i=0;i<4;++i)put(i==0?"tyre_soft":i==1?"tyre_medium":i==2?"tyre_hard":"tyre_wet",tyreCompound==i);put("battery",hybrid?battery/100:0);put("battery_percent",hybrid?battery:0);put("regen",hybrid?clampv(batteryRegen/.46,0.0,1.0):0);put("is_hybrid",hybrid);put("overtake_active",deploying);put("recharge_active",regen);put("off_track",outsideLimits);put("car_collision",carCollision);put("understeer",understeer);put("oversteer",oversteer);put("racing_line_offset",racingOffset);put("car_ahead",carAhead);put("car_ahead_distance",carAheadDistance);put("car_ahead_side",carAheadSide);put("closing_speed",closingSpeed);put("passing",passing);put("passing_side",passingSide);put("local_velocity_forward",clampv(dot(velocity,heading)/1.67,-1.0,1.0));put("local_velocity_lateral",clampv(dot(velocity,localNormal)/1.67,-1.0,1.0));put("angular_velocity",clampv(angularVelocity/6,-1.0,1.0));put("traction",traction);put("tire_slip",tireSlip);put("rpm",rpm/13000);put("rpm_value",rpm);put("gear",gear/8.0);put("gear_number",gear);put("speed_kph",speed);put("ray_left_90",rayCache[0]);put("ray_left_18",rayCache[3]);put("ray_right_18",rayCache[5]);put("ray_right_90",rayCache[8]);
        static constexpr std::array<double,4>ahead={5,10,20,40};std::array<V2,4>waypoints;for(size_t i=0;i<ahead.size();++i){waypoints[i]=track.at(projection.metres+ahead[i]).first;V2 relative=waypoints[i]-position;put(("waypoint_"+std::to_string(int(ahead[i]))+"_forward").c_str(),clampv(dot(relative,heading)/ahead[i],-1.0,1.0));put(("waypoint_"+std::to_string(int(ahead[i]))+"_right").c_str(),clampv(dot(relative,localNormal)/ahead[i],-1.0,1.0));}
        for(size_t i=0;i<3;++i){std::string prefix="opponent_"+std::to_string(i+1);put((prefix+"_forward").c_str(),opponentData[i*4]);put((prefix+"_right").c_str(),opponentData[i*4+1]);put((prefix+"_velocity_forward").c_str(),opponentData[i*4+2]);put((prefix+"_velocity_right").c_str(),opponentData[i*4+3]);put((prefix+"_present").c_str(),opponentPresence[i]);}
        put("previous_steering",steer);put("previous_throttle",throttle);put("previous_brake",brake);for(size_t i=1;i<4;++i){V2 chord=waypoints[i]-projection.point;double value=0;if(length(chord)>1e-9)value=clampv(std::atan2(cross(tangent,unit(chord)),dot(tangent,unit(chord)))*40/ahead[i],-1.0,1.0);put(("corner_curvature_"+std::to_string(int(ahead[i]))).c_str(),value);}put("corner_curvature_70",longCurvature70);put("corner_curvature_110",longCurvature110);put("corner_curvature_160",longCurvature160);put("apex_distance",apexDistance);put("apex_curvature",apexCurvature);put("target_line_offset",clampv(apexCurvature*(1-2*apexDistance)*1.1,-1.0,1.0));put("race_position",racePosition);put("field_size",fieldSize);put("position_deficit",fieldSize>1?double(racePosition-1)/(fieldSize-1):0);put("gap_to_leader_m",gapLeader);put("gap_to_next_m",gapNext);put("race_aggression",raceAggression);put("aggression_error",aggressionError);put("progress",projection.metres/std::max(1.0,track.lengthM));put("drs_eligible",drsEligible);put("drs_active",drsActive);put("drs_in_zone",drsInZone);put("drs_gap",drsGap);put("track_limits",trackLimits);put("stagnant_frames",stagnantFrames);return v;
    }
    void update(const Track& track,double dt,double now,bool traffic=false,double draft=0,double rain=0,bool damageEnabled=true,std::mt19937* rng=nullptr,bool redFlagActive=false) {
        if(!alive||finished)return;
        if(redFlagPitStopped){velocity={};speed=0;throttle=steer=0;brake=1.0;updateDrivetrain(0);return;}
        double frameDt=clampv(dt*60.0,0.0,3.0);
        if(pitTimer>0){pitTimer-=frameDt;velocity=velocity*std::pow(.72,frameDt);throttle=brake=steer=0;deploying=regen=false;batteryRegen=0;if(pitTimer<=0){wear=0;dirty=0;puncture=false;++pitstops;pitRequested=false;pitEntryCommitted=false;tyreCompound=requestedTyre;}speed=length(velocity)*216;updateDrivetrain(0);return;}
        // --- Aggression mistake system (matches Python) ---
        if(raceAggression<=0){aggressionError=0;aggressionMistakeFrames=0;aggressionMistakeCooldown=0;}
        else if(aggressionMistakeFrames>0){
            aggressionMistakeFrames=std::max(0.0,aggressionMistakeFrames-frameDt);
            if(aggressionMistakeFrames<=0 && rng){
                std::uniform_real_distribution<double> cd(90,240);
                aggressionMistakeCooldown=cd(*rng);
            }
        } else {
            aggressionError*=std::pow(.84,frameDt);
            aggressionMistakeCooldown=std::max(0.0,aggressionMistakeCooldown-frameDt);
            double mistakeProb=raceAggression*raceAggression*.0022*frameDt;
            if(rng && aggressionMistakeCooldown<=0){
                std::uniform_real_distribution<double> u01(0,1);
                if(u01(*rng)<mistakeProb){
                    double sign=u01(*rng)<.62?1.0:-1.0;
                    std::uniform_real_distribution<double> mag(.45,1.0),dur(20,55);
                    aggressionError=sign*mag(*rng);
                    aggressionMistakeFrames=dur(*rng);
                }
            }
        }
        auto inputs=controllerInputs(track,rain);carCollision=false;
        Controls controls=brain.think(inputs);
        steer=clampv(controls.steer,-1.0,1.0);throttle=clampv(controls.throttle,0.0,1.0);brake=clampv(controls.brake,0.0,1.0);
        bool pitAvailable=!track.pitBoxes().empty();
        bool emergencyPit=pitAvailable&&(wear>=.9||puncture);
        pitRequested=pitRequested||(pitAvailable&&controls.pitRequest>=.5)||emergencyPit;
        requestedTyre=clampv(controls.pitTyre,0,3);
        V2 forward{std::cos(angle),std::sin(angle)},side=normal(forward);
        // --- Pit transition state ---
        bool pitTransition=pitEntryCommitted||inPitlane||pitExitStraightFrames>0||pitExitGuidanceFrames>0;
        constexpr double PITLANE_WIDTH_M=6.0;
        // --- Wheel surfaces with pit transition tolerance (matches Python) ---
        auto wheelSurfaceAt=[&](V2 pos) -> std::string {
            std::string surf=track.surface(pos);
            if(!pitTransition) return surf;
            auto pitN=track.pitlaneNearest(pos);
            if(pitN.segment>=0 && pitN.distance<=PITLANE_WIDTH_M*.5+1.25) return "pitlane";
            if(surf=="wall" && pitN.segment>=0 && pitN.distance<=PITLANE_WIDTH_M*.5+CAR_LENGTH_M) return "grass";
            return surf;
        };
        std::array<std::string,4>surfaces{};size_t surfaceIndex=0;
        for(double longitudinal:{-1.7,1.7})for(double lateralWheel:{-1.0,1.0})surfaces[surfaceIndex++]=wheelSurfaceAt(position+forward*longitudinal+side*lateralWheel);
        auto priority=[](const std::string&s){return s=="wall"?3:s=="grass"?2:s=="kerb"?1:0;};
        std::string surface=*std::max_element(surfaces.begin(),surfaces.end(),[&](const auto&a,const auto&b){return priority(a)<priority(b);});
        double grip=surface=="asphalt"||surface=="pitlane"?1:surface=="kerb"?.78:surface=="grass"?.34:.15;
        static constexpr std::array<double,4>tyreGripBase={1.08,1,.93,.85};
        double tyreGrip=tyreGripBase[size_t(clampv(tyreCompound,0,3))];
        tyreGrip*=tyreCompound==3?.72+rain*.45:1-rain*.68;
        tyreGrip*=1-wear*.4;
        if(dirty>0){tyreGrip*=.72;dirty=std::max(0.0,dirty-frameDt);}
        if(puncture)tyreGrip*=.28;
        // --- Full pit guidance system (matches Python pit_guidance) ---
        inPitlane=track.inPitlane(position);
        V2 pitTarget{};bool havePitTarget=false;bool straightExit=false;bool recoveringFromRunoff=false;
        auto pitExitTarget=[&]() -> V2 {
            if(!track.features.contains("pit_exit")||!track.features["pit_exit"].is_number_integer()||track.points.empty())
                return track.pitCenterline.empty()?V2{}:track.pitCenterline.back();
            int exitNode=(track.features["pit_exit"].get<int>()+1)%int(track.points.size());
            size_t exitSeg=size_t(exitNode);
            double exitProgress=track.cumulative[exitSeg];
            double relative=std::fmod(exitProgress-track.startGeometryOffset+track.geometryLength,std::max(1.0,track.geometryLength));
            double exitMetres=relative*track.lengthM/std::max(1.0,track.geometryLength);
            double guidanceElapsed=std::max(0.0,90.0-pitExitGuidanceFrames);
            return track.at(exitMetres+guidanceElapsed*.55).first;
        };
        if(pitExitStraightFrames>0){
            pitExitStraightFrames=std::max(0.0,pitExitStraightFrames-frameDt);
            pitEntryCommitted=false;straightExit=true;
        } else if(pitExitGuidanceFrames>0){
            pitExitGuidanceFrames=std::max(0.0,pitExitGuidanceFrames-frameDt);
            pitEntryCommitted=false;pitTarget=pitExitTarget();havePitTarget=true;
        } else if(inPitlane && (pitRequested || pitExitStraightFrames>0 || pitExitGuidanceFrames>0 || pitEntryCommitted)){
            pitEntryCommitted=false;
            if(!track.pitCenterline.empty() && length(position-track.pitCenterline.back())<=18){
                pitExitStraightFrames=30;pitExitGuidanceFrames=90;straightExit=true;
            } else {
                double lookahead=clampv(length(velocity)*32,10.0,24.0);
                pitTarget=track.pitlanePointAhead(position,lookahead);havePitTarget=true;
            }
        } else if(pitRequested && !track.pitCenterline.empty()){
            double entry=track.featureMetres("pit_entry");
            if(entry>=0){
                double entryDist=std::fmod(entry-s+track.lengthM,track.lengthM);
                if(entryDist<=130) pitEntryCommitted=true;
                if(pitEntryCommitted){
                    if(entryDist<=130){
                        double lookahead=clampv(length(velocity)*20,14.0,34.0);
                        V2 mainTarget=track.at(s+std::min(lookahead,std::max(entryDist,0.0))).first;
                        double merge=clampv((55-entryDist)/45,0.0,1.0);
                        V2 pitMerge=track.pitCenterline[0]+(track.pitCenterline[std::min(size_t(1),track.pitCenterline.size()-1)]-track.pitCenterline[0])*clampv((merge-.62)/.38,0.0,1.0);
                        pitTarget=mainTarget+(pitMerge-mainTarget)*merge;havePitTarget=true;
                    } else {
                        pitTarget=track.pitCenterline[std::min(size_t(1),track.pitCenterline.size()-1)];havePitTarget=true;
                    }
                }
            }
        } else { pitEntryCommitted=false; }
        recoveringFromRunoff=pitTransition && track.surface(position)!="asphalt" && track.surface(position)!="kerb" && track.surface(position)!="pitlane";
        if(recoveringFromRunoff){
            if(straightExit){havePitTarget=false;}
            else if(pitExitGuidanceFrames>0){pitTarget=pitExitTarget();havePitTarget=true;}
            else if(inPitlane){pitTarget=track.pitlanePointAhead(position,8);havePitTarget=true;}
            else if(!track.pitCenterline.empty()){pitTarget=track.pitCenterline[std::min(size_t(1),track.pitCenterline.size()-1)];havePitTarget=true;}
        }
        // Apply pit speed and steering guidance
        if(havePitTarget || straightExit || pitExitGuidanceFrames>0 || pitExitStraightFrames>0 || (pitEntryCommitted && pitRequested) || (inPitlane && pitRequested)){
            double spd=length(velocity),targetSpeed=80/216.0*(inPitlane?.82:.95);
            if(spd>targetSpeed){throttle=0;brake=std::max(brake,clampv((spd-targetSpeed)/std::max(80/216.0,1e-9),.18,1.0));}
            else if(straightExit){brake=0;throttle=clampv(std::max(throttle,.42),0.0,.58);}
            else if(recoveringFromRunoff){brake=0;throttle=clampv(std::max(throttle,.62),0.0,.72);}
            else if(inPitlane){brake=0;throttle=std::max(throttle,.34);}
            else {brake=0;throttle=clampv(std::max(throttle,.34),0.0,.55);}
            if(straightExit){steer=0;}
            else if(havePitTarget){
                V2 desired=pitTarget-position;
                if(length(desired)<=1e-9 && inPitlane && track.pitCenterline.size()>=2)
                    desired=track.pitCenterline.back()-track.pitCenterline[track.pitCenterline.size()-2];
                if(length(desired)>1e-9){
                    desired=unit(desired);
                    double steerError=std::atan2(cross(forward,desired),dot(forward,desired))/PI;
                    double gain=recoveringFromRunoff?3.45:(inPitlane||pitExitGuidanceFrames>0)?2.75:2.35;
                    steer=clampv(steerError*gain-angularVelocity*.10,-1.0,1.0);
                }
            }
        }
        double rawSpeed=length(velocity);slipstream=clampv(draft,0.0,1.0);
        regen=hybrid&&controls.recharge>=.5&&battery<100;
        double hybridDrs=hybrid&&drsEligible&&!regen&&!inPitlane&&battery>0;
        drsActive=hybridDrs||(!hybrid&&drsEligible&&drsInZone&&!inPitlane);
        deploying=hybrid&&!regen&&controls.overtake>=.5&&battery>0&&throttle>0&&brake<.05;
        double electric=std::max(double(deploying),hybridDrs),power=hybrid?(regen?1.1875*.70:1.1875*(.80+.20*electric+.30*hybridDrs)):1;
        batteryRegen=0;if(hybrid){batteryRegen=std::pow(brake,.85)*clampv(rawSpeed,0.0,1.0)*.46*frameDt;if(regen)batteryRegen+=(5.5/60)*throttle*frameDt;battery=clampv(battery+batteryRegen-(electric*.08+hybridDrs*.04)*throttle*frameDt,0.0,100.0);}else{battery=0;deploying=regen=false;}
        double maximum=(hybrid?(regen?1.48:1.55+.12*electric+.06*hybridDrs):(drsActive?1.72:1.67))*(1-fuel*.0015)*(1+draft*.075);
        double aeroGrip=1-draft*.22,frontGrip=clampv(grip*tyreGrip*aeroGrip,.05,1.15),cornerLoad=std::abs(steer)*rawSpeed;
        double underTarget=clampv((cornerLoad-frontGrip*1.28)*.30+draft*std::abs(steer)*.04,0.0,1.0);
        double overTarget=clampv((cornerLoad*throttle*std::max(0.0,1.05-grip*tyreGrip)-.16)*1.35+(puncture?.35:0),0.0,1.0);
        understeer+=(underTarget-understeer)*clampv((underTarget>understeer?.14:.26)*frameDt,0.0,1.0);
        oversteer+=(overTarget-oversteer)*clampv(.18*frameDt,0.0,1.0);
        double oldAngle=angle;
        angle+=steer*(3.68+rawSpeed*.79)*PI/180*grip*tyreGrip*aeroGrip*(1-understeer*.12+oversteer*.20)*frameDt;
        updateDrivetrain(throttle);
        static constexpr std::array<double,8>ratios={3.20,2.55,2.08,1.72,1.45,1.25,1.16,.94};
        double launch=.52+.48*clampv(rawSpeed/.60,0.0,1.0),tyreTraction=clampv(1+(tyreGrip-1)*.40,.30,1.05),available=clampv(launch*tyreTraction,.10,1.0);
        velocity=velocity+forward*(throttle*(1-brake)*.0086*drivetrainPower()*power*(1+draft*.08)*grip*available*frameDt);
        double forwardSpeed=std::max(0.0,dot(velocity,forward)),brakingDelta=std::min(forwardSpeed,brake*.028*grip*tyreGrip*frameDt);
        velocity=velocity-forward*brakingDelta;
        double engineRequest=clampv((.25-throttle)/.25,0.0,1.0),ratioStrength=ratios[size_t(gear-1)]/ratios[0],engineDelta=std::min(std::max(0.0,dot(velocity,forward)),engineRequest*(.0012+.0028*ratioStrength)*grip*frameDt);
        velocity=velocity-forward*engineDelta;
        V2 lateralVelocity=velocity-forward*dot(velocity,forward);
        velocity=velocity-lateralVelocity*(clampv(grip*tyreGrip*.24*(1-draft*.28)*(1-oversteer*.62),.025,.27)*frameDt);
        double current=length(velocity),scrub=std::min(current,std::pow(std::abs(steer),1.7)*current*current*.004*(1+understeer*.45+oversteer*.65)*frameDt);
        if(current>1e-9&&scrub>0)velocity=unit(velocity)*std::max(0.0,current-scrub);
        current=length(velocity);double drag=std::min(current,(.00035+current*current*.00056*((!hybrid&&drsActive)?.30:1))*frameDt);
        if(current>1e-9&&drag>0)velocity=unit(velocity)*std::max(0.0,current-drag);
        if(length(velocity)>maximum)velocity=unit(velocity)*maximum;
        // --- Grass friction: pit transition gets milder deceleration (matches Python) ---
        if(std::find(surfaces.begin(),surfaces.end(),"grass")!=surfaces.end()){
            if(pitTransition){dirty=std::max(dirty,60.0);velocity=velocity*std::pow(.992,frameDt);}
            else {dirty=180;velocity=velocity*std::pow(.975,frameDt);}
        }
        bool allOut=std::all_of(surfaces.begin(),surfaces.end(),[](const auto&value){return value!="asphalt"&&value!="kerb"&&value!="pitlane";});
        if(allOut)offTrackFrames+=frameDt;
        // --- Track limits counter (matches Python) ---
        if(allOut&&!outsideLimits)++trackLimits;
        outsideLimits=allOut;
        // --- Wall recovery: pit recovery tolerance + proper car clearance (matches Python) ---
        bool pitRecovery=pitTransition && track.pitlaneNearest(position).distance<=PITLANE_WIDTH_M*.5+CAR_LENGTH_M*1.5;
        if(surface=="wall" && pitRecovery){
            velocity=velocity*.72;
        } else if(surface=="wall"){
            if(damageEnabled)health-=rawSpeed*7;
            velocity=velocity*-.2;
            auto p=track.project(position);
            V2 direction=unit(position-p.point);
            double carClearance=std::sqrt(CAR_LENGTH_M*CAR_LENGTH_M*.25+CAR_WIDTH_M*CAR_WIDTH_M*.25);
            double safeDistance=std::max(track.widthAtProjection(p)*.5+.5,15.0-carClearance-.5);
            position=p.point+direction*safeDistance;
        }
        V2 previousPosition=position;position=position+velocity*frameDt;
        inPitlane=track.inPitlane(position);
        if(inPitlane&&length(velocity)>80/216.0)velocity=unit(velocity)*(80/216.0);
        angularVelocity=(angle-oldAngle)*180/PI/std::max(frameDt,1e-9);
        V2 newHeading{std::cos(angle),std::sin(angle)};
        double lateralSpeed=std::abs(dot(velocity,normal(newHeading))),velocitySlip=lateralSpeed/std::max(length(velocity),.08);
        tireSlip=clampv(velocitySlip*1.15+understeer*.12+oversteer*.48,0.0,1.0);traction=1-tireSlip;
        updateDrivetrain(throttle);
        // --- Wear with aggression scaling + probabilistic puncture (matches Python) ---
        static constexpr std::array<double,4>wearRates={.0018,.0012,.0008,.0025};
        double wearRate=wearRates[size_t(clampv(tyreCompound,0,3))]*(tyreCompound==3&&rain<.2?3.6:1);
        wearRate*=1.0+raceAggression*.12;
        double wearPercent=wear*100;wearPercent+=rawSpeed*wearRate*frameDt;
        wear=clampv(wearPercent/100,0.0,1.2);
        fuel=std::max(0.0,fuel-throttle*(1-brake)*.0009*frameDt);
        // Probabilistic puncture (matches Python: wear>70% triggers random puncture chance)
        if(!puncture && wearPercent>70 && rng){
            double punctureChance=std::pow((wearPercent-70)/30,4)*.0005;
            std::uniform_real_distribution<double> u01(0,1);
            if(u01(*rng)<punctureChance) puncture=true;
        }
        auto projection=track.project(position);double newS=projection.metres;
        double delta=previousProgress<0?0:std::fmod(newS-previousProgress+track.lengthM*1.5,track.lengthM)-track.lengthM*.5;
        double physical=length(position-previousPosition);
        if(std::abs(delta)>std::max(.05,physical*1.5))delta=clampv(dot(position-previousPosition,forward),-physical,physical);
        raceDistance+=delta;forwardDistance+=std::max(0.0,delta);reverseDistance+=std::max(0.0,-delta);
        // --- Stagnant frames tracking (matches Python) ---
        if(inPitlane||pitTimer>0||redFlagPitStopped)stagnantFrames=0;
        else if(rawSpeed<.08)stagnantFrames+=frameDt;else stagnantFrames=0;
        double stalledPenalty=std::min(std::max(0.0,stagnantFrames-180)*.025,100.0);
        // --- Full fitness formula (matches Python) ---
        fitness=forwardDistance+overtakes*150-reverseDistance*2-trackLimits*20.0-offTrackFrames*.04-(100-health)*1.5-controlPenalty-collisionPenalty-stalledPenalty;
        s=newS;lateral=projection.lateral;
        if(previousProgress>=0){
            for(const auto&key:std::array<const char*,3>{"drs_detection","drs_entry","drs_exit"}){double marker=track.featureMetres(key);if(crossed(previousProgress,s,marker,track.lengthM)){if(std::string(key)=="drs_detection")drsEligible=drsGap<1;if(std::string(key)=="drs_entry")drsInZone=true;if(std::string(key)=="drs_exit"){drsInZone=false;if(!hybrid)drsActive=false;}}}
            std::vector<double>sectors;if(track.features.contains("sectors")&&track.features["sectors"].is_array())for(const auto&value:track.features["sectors"])if(value.is_number_integer()){size_t index=size_t(clampv(value.get<int>(),0,int(track.points.size())-1));double geometry=track.cumulative[index],relative=std::fmod(geometry-track.startGeometryOffset+track.geometryLength,std::max(1.0,track.geometryLength));sectors.push_back(relative*track.lengthM/std::max(1.0,track.geometryLength));}std::sort(sectors.begin(),sectors.end());if(checkpoint<int(sectors.size())&&crossed(previousProgress,s,sectors[size_t(checkpoint)],track.lengthM))++checkpoint;bool finishCrossed=track.crossedTimingLine(previousPosition,position)||(inPitlane&&track.crossedTimingLine(previousPosition,position,true));if(finishCrossed&&(sectors.empty()||checkpoint>=int(sectors.size()))){if(!redFlagActive){++lap;++tyreLaps;lastLap=now-lapStart;lapStart=now;bestLap=std::min(bestLap,lastLap);}checkpoint=0;}
        }
        previousProgress=s;speed=length(velocity)*216;
        auto boxes=track.pitBoxes();
        if(pitRequested&&inPitlane&&!boxes.empty()){
            size_t box=size_t(startingPosition-1)%boxes.size();
            if(length(position-boxes[box])<6){
                if(redFlagActive){redFlagPitStopped=true;velocity={};speed=0;throttle=steer=0;brake=1.0;updateDrivetrain(0);return;}
                pitTimer=120;velocity=velocity*.25;
            }
        } else if(redFlagActive&&inPitlane&&track.pitBoxes().empty()&&length(velocity)<1.2){
            redFlagPitStopped=true;velocity={};speed=0;throttle=steer=0;brake=1.0;updateDrivetrain(0);return;
        } else if(redFlagActive&&track.pitlanePoints.empty()&&s<60&&length(velocity)<1.2){
            redFlagPitStopped=true;velocity={};speed=0;throttle=steer=0;brake=1.0;updateDrivetrain(0);return;
        }
        alive=health>0;poseReady=true;
    }
};

enum class Mode { Menu, TrackEditor, Algorithm, Training, RaceSetup, Race, HotlapSetup, Hotlap, ReplaySetup, Replay };

struct ReplayCar { std::string name,generation{"ICE"};double x{},y{},angle{},speed{},battery{},fuel{},wear{},rpm{},health{100},throttle{},brake{},slipstream{};int lap{},gear{1},pitstops{},tyre{};bool pitRequested{},overtake{},recharge{},drsEligible{},drsActive{},removed{};SDL_Color col{60,210,180,255}; };
struct ReplayFrame { double time{}; std::vector<ReplayCar> cars; };
struct RaceEntry { std::string name; int brainIndex{-1},colorIndex{}; float fuel{55}; int tyre{}; };

class App {
    SDL_Window* win{}; SDL_Renderer* ren{}; bool running{true}; Mode mode{Mode::Menu};
    SDL_Texture* iceCarMaster{}; SDL_Texture* hybridCarMaster{};
    ImFont* uiFont{};ImFont* monoFont{};
    std::vector<fs::path> trackFiles,brainFiles,replayFiles; size_t trackIndex{},brainIndex{},replayIndex{};
    Track track; std::vector<Car> cars; Brain chosenBrain;bool chosenBrainEmpty{};std::mt19937 rng{std::random_device{}()};
    bool paused{},showFps{}; double simTime{},countdown{},cameraZoom{DEFAULT_CAMERA_ZOOM},currentFps{}; int focus{}; std::string notice;
    SDL_Rect logicalViewport{0,0,W,H}; double logicalScale{1.0};
    double editorZoom{1.0}; V2 editorPan{}; Transform editorBase{}; int draggedNode{-1};
    int draggedPitNode{-1},selectedEditorNode{-1};bool selectedEditorPit{};std::string editorTool{"route"};
    bool editorPanning{}; int lastMouseX{},lastMouseY{};
    bool hybrid{true},racecraft{}; int population{20},raceCars{20},generation{},targetLaps{3}; double bestFitness{-1}; Brain champion;
    int timingMetric{0};
    std::vector<RaceEntry> raceEntries;int selectedRaceEntry{};bool raceHybrid{true},raceTeams{};int raceWeather{};
    double rainLevel{};
    int lastWeatherLap{-1};double weatherForecast{.5},flagTimer{};std::string flagState{"GREEN"};
    bool redFlagActive{false},redFlagSuspended{false};int redFlagLap{1};std::vector<int> redFlagSnapshotOrder;int selectedRedFlagEntry{0};
    std::array<std::string,10> raceTeamNames{};
    int hotlapBrainIndex{-1};bool hotlapHybrid{true};
    std::string editorSource; size_t cursor{},anchor{}; bool selecting{}; int editorScroll{}; std::vector<std::string> undo,redo;
    std::vector<ReplayFrame> replay; double replayTime{},replaySpeed{1};
    std::vector<ReplayFrame> capturedReplay;double replayCaptureAccumulator{};bool replaySaved{};
    std::array<SDL_Color,20> palette{{rgb(0x3ee0c1),rgb(0x4aa3ff),rgb(0xff7f3f),rgb(0xb267ff),rgb(0xffcf3f),rgb(0x36c66f),rgb(0xff5464),rgb(0x78d8ff),rgb(0xff8fc7),rgb(0xa4df4b),rgb(0xe67e22),rgb(0x9b59b6),rgb(0x1abc9c),rgb(0xe74c3c),rgb(0x3498db),rgb(0xf1c40f),rgb(0x2ecc71),rgb(0xecf0f1),rgb(0x95a5a6),rgb(0xd35400)}};
    std::array<SDL_Texture*,5> menuBgTextures{};
public:
    ~App(){ if(ImGui::GetCurrentContext()){ImGui_ImplSDLRenderer2_Shutdown();ImGui_ImplSDL2_Shutdown();ImGui::DestroyContext();}for(SDL_Texture* t:menuBgTextures)if(t)SDL_DestroyTexture(t);if(iceCarMaster)SDL_DestroyTexture(iceCarMaster);if(hybridCarMaster)SDL_DestroyTexture(hybridCarMaster);if(ren)SDL_DestroyRenderer(ren);if(win)SDL_DestroyWindow(win);SDL_Quit(); }
    SDL_Texture* loadTexture(const fs::path& path){
        int width=0,height=0,channels=0;unsigned char* pixels=stbi_load(path.string().c_str(),&width,&height,&channels,4);
        if(!pixels)return nullptr;
        SDL_Surface* surface=SDL_CreateRGBSurfaceWithFormatFrom(pixels,width,height,32,width*4,SDL_PIXELFORMAT_ABGR8888);
        if(!surface){stbi_image_free(pixels);return nullptr;}
        SDL_Texture* texture=SDL_CreateTextureFromSurface(ren,surface);
        SDL_FreeSurface(surface);
        stbi_image_free(pixels);
        if(texture)SDL_SetTextureBlendMode(texture,SDL_BLENDMODE_BLEND);
        return texture;
    }
    SDL_Texture* loadCarMaster(const fs::path& path){
        int width=0,height=0,channels=0;unsigned char* pixels=stbi_load(path.string().c_str(),&width,&height,&channels,4);
        if(!pixels){std::cerr<<"Could not load car model: "<<path<<'\n';return nullptr;}
        int minX=width,minY=height,maxX=-1,maxY=-1;
        for(int y=0;y<height;++y)for(int x=0;x<width;++x)if(pixels[(y*width+x)*4+3]>=8){minX=std::min(minX,x);minY=std::min(minY,y);maxX=std::max(maxX,x);maxY=std::max(maxY,y);}
        if(maxX<minX){stbi_image_free(pixels);return nullptr;}
        const int croppedWidth=maxX-minX+1,croppedHeight=maxY-minY+1;
        std::vector<unsigned char> cropped(size_t(croppedWidth)*size_t(croppedHeight)*4);
        for(int y=0;y<croppedHeight;++y)std::copy_n(pixels+((y+minY)*width+minX)*4,size_t(croppedWidth)*4,cropped.data()+size_t(y)*size_t(croppedWidth)*4);
        stbi_image_free(pixels);
        SDL_Surface* surface=SDL_CreateRGBSurfaceWithFormatFrom(cropped.data(),croppedWidth,croppedHeight,32,croppedWidth*4,SDL_PIXELFORMAT_ABGR8888);
        if(!surface)return nullptr;SDL_Texture* texture=SDL_CreateTextureFromSurface(ren,surface);SDL_FreeSurface(surface);
        if(texture)SDL_SetTextureBlendMode(texture,SDL_BLENDMODE_BLEND);return texture;
    }
    bool init() {
        if(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER)!=0){std::cerr<<SDL_GetError()<<'\n';return false;}
        SDL_DisplayMode desktop{};SDL_GetDesktopDisplayMode(0,&desktop);
        int initialW=std::min(W,std::max(1100,desktop.w-120));
        int initialH=std::min(H,std::max(680,desktop.h-140));
        win=SDL_CreateWindow("Formula AI Lab - Native C++",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,initialW,initialH,SDL_WINDOW_RESIZABLE|SDL_WINDOW_ALLOW_HIGHDPI);
        ren=SDL_CreateRenderer(win,-1,SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
        if(!ren)ren=SDL_CreateRenderer(win,-1,SDL_RENDERER_SOFTWARE); if(!ren)return false;
        SDL_SetWindowMinimumSize(win,1000,620);SDL_SetRenderDrawBlendMode(ren,SDL_BLENDMODE_BLEND);
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY,"linear");
        IMGUI_CHECKVERSION(); ImGui::CreateContext();
        ImGuiIO& io=ImGui::GetIO();io.ConfigFlags|=ImGuiConfigFlags_NavEnableKeyboard;io.IniFilename=nullptr;
        if(fs::exists("/System/Library/Fonts/SFNS.ttf"))uiFont=io.Fonts->AddFontFromFileTTF("/System/Library/Fonts/SFNS.ttf",17.5f);
        if(fs::exists("/System/Library/Fonts/SFNSMono.ttf"))monoFont=io.Fonts->AddFontFromFileTTF("/System/Library/Fonts/SFNSMono.ttf",16.0f);
        if(!uiFont){ImFontConfig fontConfig;fontConfig.SizePixels=17.5f;uiFont=io.Fonts->AddFontDefault(&fontConfig);}if(!monoFont)monoFont=uiFont;io.FontDefault=uiFont;
        ImGui::StyleColorsDark();styleImGui();
        ImGui_ImplSDL2_InitForSDLRenderer(win,ren);ImGui_ImplSDLRenderer2_Init(ren);
        iceCarMaster=loadCarMaster(fs::path(F1RACE_CAR_ASSET_DIR)/"ice_2000s_master.png");
        hybridCarMaster=loadCarMaster(fs::path(F1RACE_CAR_ASSET_DIR)/"hybrid_2022_master.png");
        const std::array<const char*,5> menuImages={"track_studio.jpg","ai_training.jpg","race_weekend.jpg","hotlap_clock.jpg","replay_theatre.jpg"};
        for(size_t i=0;i<5;++i){
            fs::path p=fs::path(F1RACE_CAR_ASSET_DIR).parent_path()/"menu"/menuImages[i];
            menuBgTextures[i]=loadTexture(p);
        }
        refreshFiles(); loadTrack(0); loadBrain(0);ensureRaceEntries(); return true;
    }
    void styleImGui(){
        ImGuiStyle& s=ImGui::GetStyle();s.WindowRounding=10;s.ChildRounding=8;s.FrameRounding=6;s.PopupRounding=8;s.ScrollbarRounding=8;s.GrabRounding=5;s.WindowPadding={14,12};s.FramePadding={8,5};s.ItemSpacing={8,6};s.CellPadding={6,4};s.WindowBorderSize=1;s.FrameBorderSize=0;
        auto& c=s.Colors;c[ImGuiCol_WindowBg]=ImVec4(.035f,.09f,.105f,.97f);c[ImGuiCol_ChildBg]=ImVec4(.055f,.15f,.155f,.96f);c[ImGuiCol_PopupBg]=ImVec4(.035f,.09f,.105f,.99f);c[ImGuiCol_Border]=ImVec4(.15f,.36f,.36f,1);c[ImGuiCol_Text]=ImVec4(.90f,.95f,.93f,1);c[ImGuiCol_TextDisabled]=ImVec4(.50f,.62f,.60f,1);c[ImGuiCol_Button]=ImVec4(.07f,.22f,.22f,1);c[ImGuiCol_ButtonHovered]=ImVec4(.10f,.34f,.32f,1);c[ImGuiCol_ButtonActive]=ImVec4(.12f,.48f,.42f,1);c[ImGuiCol_Header]=ImVec4(.08f,.28f,.28f,1);c[ImGuiCol_HeaderHovered]=ImVec4(.12f,.38f,.36f,1);c[ImGuiCol_FrameBg]=ImVec4(.035f,.12f,.14f,1);c[ImGuiCol_FrameBgHovered]=ImVec4(.06f,.20f,.22f,1);c[ImGuiCol_CheckMark]=ImVec4(.27f,.88f,.76f,1);c[ImGuiCol_SliderGrab]=ImVec4(.27f,.88f,.76f,1);c[ImGuiCol_NavCursor]=ImVec4(.30f,.70f,1,1);
    }
    void refreshFiles(){trackFiles=filesFor("tracks",".json");brainFiles=filesFor("brains",".json");replayFiles=filesFor("replays",".json");}
    void ensureRaceEntries(){static const std::array<const char*,20> names={"NOVA","APEX","VOLT","ZENITH","ORBIT","PULSE","COMET","ECHO","BLAZE","KITE","ONYX","RAPTOR","SOLAR","DRIFT","TITAN","FLUX","VEGA","STORM","RUNE","FROST"};while(raceEntries.size()<20){size_t i=raceEntries.size();RaceEntry e;e.name=names[i];e.colorIndex=int(i);e.brainIndex=brainFiles.empty()?-2:int(i%brainFiles.size());e.tyre=int(i%3);raceEntries.push_back(e);}for(size_t i=0;i<raceTeamNames.size();++i)if(raceTeamNames[i].empty())raceTeamNames[i]="TEAM "+std::to_string(i+1);}
    void loadTrack(size_t i){if(trackFiles.empty()){makeStarter();return;}trackIndex=i%trackFiles.size();if(!track.load(trackFiles[trackIndex]))makeStarter();}
    void makeStarter(){track.name="Starter Ring";track.points.clear();track.widths.clear();track.grassWidths.clear();track.pitlanePoints.clear();track.pitlaneWidths.clear();track.pitlaneGrassWidths.clear();track.features={{"start_finish",0},{"pit_start_finish",nullptr},{"sectors",json::array()},{"pit_entry",nullptr},{"pit_exit",nullptr},{"pit_boxes",json::array()},{"drs_detection",nullptr},{"drs_entry",nullptr},{"drs_exit",nullptr}};for(int i=0;i<96;++i){double a=i*2*PI/96;track.points.push_back({450+std::cos(a)*310,450+std::sin(a)*210});track.widths.push_back(12);track.grassWidths.push_back(28);}track.rebuild();}
    void loadBrain(size_t i){if(brainFiles.empty()){chosenBrain=Brain{};chosenBrainEmpty=true;return;}brainIndex=i%brainFiles.size();chosenBrain=Brain::load(brainFiles[brainIndex]);chosenBrainEmpty=false;}
    void selectEmptyBrain(){chosenBrain=Brain{};chosenBrainEmpty=true;}
    void run(int frameLimit=-1){uint64_t then=SDL_GetPerformanceCounter();int frames=0;while(running&&(frameLimit<0||frames++<frameLimit)){uint64_t now=SDL_GetPerformanceCounter();double dt=clampv(double(now-then)/SDL_GetPerformanceFrequency(),0.0,.05);then=now;double measured=dt>1e-6?1.0/dt:0.0;if(measured>0)currentFps=currentFps>0?currentFps*.90+measured*.10:measured;events();update(dt);draw();}}
    void resizeForTest(int width,int height){SDL_SetWindowSize(win,width,height);}
    bool saveScreenshot(const fs::path& path){int rw=0,rh=0;SDL_GetRendererOutputSize(ren,&rw,&rh);SDL_Surface* shot=SDL_CreateRGBSurfaceWithFormat(0,rw,rh,32,SDL_PIXELFORMAT_ARGB8888);if(!shot)return false;bool ok=SDL_RenderReadPixels(ren,nullptr,SDL_PIXELFORMAT_ARGB8888,shot->pixels,shot->pitch)==0&&SDL_SaveBMP(shot,path.string().c_str())==0;SDL_FreeSurface(shot);return ok;}
    void updateLogicalViewport(){int rw=0,rh=0;SDL_GetRendererOutputSize(ren,&rw,&rh);logicalScale=std::min(double(rw)/W,double(rh)/H);logicalViewport={int((rw-W*logicalScale)/2),int((rh-H*logicalScale)/2),int(W*logicalScale),int(H*logicalScale)};}
    std::pair<int,int> logicalMouse(int windowX,int windowY){int ww=1,wh=1,rw=1,rh=1;SDL_GetWindowSize(win,&ww,&wh);SDL_GetRendererOutputSize(ren,&rw,&rh);double px=windowX*double(rw)/std::max(1,ww),py=windowY*double(rh)/std::max(1,wh);return {int((px-logicalViewport.x)/logicalScale),int((py-logicalViewport.y)/logicalScale)};}
    void events(){SDL_Event e;while(SDL_PollEvent(&e)){ImGui_ImplSDL2_ProcessEvent(&e);if(e.type==SDL_QUIT){running=false;continue;}updateLogicalViewport();int windowX,windowY;SDL_GetMouseState(&windowX,&windowY);auto[mx,my]=logicalMouse(windowX,windowY);bool keyboard=ImGui::GetIO().WantCaptureKeyboard,mouse=ImGui::GetIO().WantCaptureMouse;if(e.type==SDL_KEYDOWN){bool cmd=e.key.keysym.mod&(KMOD_CTRL|KMOD_GUI);if(mode==Mode::Algorithm&&e.key.keysym.sym==SDLK_ESCAPE){mode=Mode::Menu;continue;}if(mode==Mode::Algorithm&&cmd&&e.key.keysym.sym==SDLK_s){saveAlgorithm();continue;}if(!keyboard)key(e.key);}if(!mouse&&e.type==SDL_MOUSEBUTTONDOWN)mouseDown(e.button,mx,my);if(e.type==SDL_MOUSEBUTTONUP){selecting=false;draggedNode=-1;draggedPitNode=-1;editorPanning=false;}if(!mouse&&e.type==SDL_MOUSEMOTION)mouseMotion(mx,my);if(!mouse&&e.type==SDL_MOUSEWHEEL)wheel(e.wheel,mx,my);}}
    void key(const SDL_KeyboardEvent& k){
        SDL_Keycode key=k.keysym.sym;bool cmd=(k.keysym.mod&(KMOD_CTRL|KMOD_GUI));bool shift=k.keysym.mod&KMOD_SHIFT;
        if(mode==Mode::Algorithm){editorKey(key,cmd,shift);return;}
        if(key==SDLK_l&&!k.repeat)showFps=!showFps;
        if(key==SDLK_ESCAPE){if(mode==Mode::Menu)running=false;else{mode=Mode::Menu;cars.clear();paused=false;}return;}
        if(mode==Mode::Menu&&key>=SDLK_1&&key<=SDLK_5){openWorkspace(int(key-SDLK_1));return;}
        if(mode==Mode::Menu&&key==SDLK_LEFT&&!trackFiles.empty()){loadTrack((trackIndex+trackFiles.size()-1)%trackFiles.size());return;}
        if(mode==Mode::Menu&&key==SDLK_RIGHT&&!trackFiles.empty()){loadTrack((trackIndex+1)%trackFiles.size());return;}
        if(mode==Mode::Menu&&key==SDLK_b&&!brainFiles.empty()){loadBrain((brainIndex+brainFiles.size()-1)%brainFiles.size());return;}
        if(mode==Mode::Menu&&key==SDLK_n&&!brainFiles.empty()){loadBrain((brainIndex+1)%brainFiles.size());return;}
        if(mode==Mode::Menu&&key==SDLK_g){hybrid=!hybrid;notice=hybrid?"HYBRID SELECTED":"ICE SELECTED";return;}
        if(mode==Mode::HotlapSetup){const int choiceCount=int(brainFiles.size())+1;int shifted=hotlapBrainIndex+1;if(key==SDLK_LEFT||key==SDLK_UP)shifted=(shifted+choiceCount-1)%choiceCount;if(key==SDLK_RIGHT||key==SDLK_DOWN)shifted=(shifted+1)%choiceCount;hotlapBrainIndex=shifted-1;if(key==SDLK_RETURN||key==SDLK_SPACE)startHotlap();return;}
        if(mode==Mode::RaceSetup&&key==SDLK_RETURN){startRace();return;}
        if(key==SDLK_SPACE)paused=!paused;
        int cameraCount=int(cars.size());
        if(mode==Mode::Replay&&!replay.empty()){
            auto it=std::lower_bound(replay.begin(),replay.end(),replayTime,[](const ReplayFrame&f,double v){return f.time<v;});
            cameraCount=int((it==replay.end()?replay.back():*it).cars.size());
        }
        if(key==SDLK_UP&&cameraCount)focus=(focus-1+cameraCount)%cameraCount;
        if(key==SDLK_DOWN&&cameraCount)focus=(focus+1)%cameraCount;
        if(key==SDLK_LEFTBRACKET)cameraZoom=std::max(MIN_CAMERA_ZOOM,cameraZoom/1.15);
        if(key==SDLK_RIGHTBRACKET)cameraZoom=std::min(MAX_CAMERA_ZOOM,cameraZoom*1.15);
        if(mode==Mode::Training||mode==Mode::Race||mode==Mode::Hotlap){if(key==SDLK_LEFT)timingMetric=(timingMetric+8)%9;if(key==SDLK_RIGHT)timingMetric=(timingMetric+1)%9;}
        if(mode==Mode::TrackEditor){if(key==SDLK_c)clearTrackEditor();if(key==SDLK_s)saveTrack();if(key==SDLK_0)editorTool="pit_finish";if(key>=SDLK_1&&key<=SDLK_9){static const std::array<const char*,9> tools={"route","kerb","sector","start","pit_entry","pit_exit","pitlane","pit_box","delete"};editorTool=tools[size_t(key-SDLK_1)];}if(key==SDLK_MINUS||key==SDLK_KP_MINUS)track.adjustAllWidths(-0.5);if(key==SDLK_EQUALS||key==SDLK_KP_PLUS)track.adjustAllWidths(0.5);if(key==SDLK_LEFTBRACKET)track.adjustAllGrassWidths(-2.0);if(key==SDLK_RIGHTBRACKET)track.adjustAllGrassWidths(2.0);}
        if(mode==Mode::Training){if(key==SDLK_r)evolve();if(key==SDLK_s)saveChampion();if(key==SDLK_EQUALS||key==SDLK_KP_PLUS)changePopulation(1);if(key==SDLK_MINUS||key==SDLK_KP_MINUS)changePopulation(-1);}
        if(mode==Mode::Race&&key==SDLK_r)triggerRedFlag();
        if(mode==Mode::Race&&key==SDLK_s)saveRaceReplay();
        if(mode==Mode::Race&&key==SDLK_w&&raceWeather==2)rainLevel=rainLevel>.5?0:1;
        if(mode==Mode::Race&&key==SDLK_p&&!cars.empty())cars[size_t(clampv(focus,0,int(cars.size())-1))].wear=1.0;
        if(mode==Mode::Hotlap&&key==SDLK_r)startHotlap();
        if(mode==Mode::Replay){if(key==SDLK_j)replaySpeed=-2;if(key==SDLK_k){replaySpeed=0;paused=true;}if(key==SDLK_l){replaySpeed=2;paused=false;}if(key==SDLK_LEFT)replayTime=std::max(0.0,replayTime-5);if(key==SDLK_RIGHT&&!replay.empty())replayTime=std::min(replay.back().time,replayTime+5);}
    }
    void openWorkspace(int index){
        if(index==0){mode=Mode::TrackEditor;editorZoom=1;editorPan={};editorBase=fitTrack(track,{30,115,1180,745});draggedNode=-1;draggedPitNode=-1;selectedEditorNode=-1;selectedEditorPit=false;editorTool="route";return;}
        if(index==1){mode=Mode::Algorithm;loadAlgorithm();return;}
        if(index==2){ensureRaceEntries();mode=Mode::RaceSetup;return;}
        if(index==3){hotlapBrainIndex=brainFiles.empty()?-1:int(std::min(brainIndex,brainFiles.size()-1));hotlapHybrid=hybrid;mode=Mode::HotlapSetup;return;}
        mode=Mode::ReplaySetup;
    }
    static bool inside(int x,int y,SDL_Rect r){return x>=r.x&&x<r.x+r.w&&y>=r.y&&y<r.y+r.h;}
    Transform editorTransform()const{SDL_Rect area{30,115,1180,745};Transform t=editorBase;double cx=area.x+area.w*.5,cy=area.y+area.h*.5;t.ox=cx+(t.ox-cx)*editorZoom+editorPan.x;t.oy=cy+(t.oy-cy)*editorZoom+editorPan.y;t.scale*=editorZoom;return t;}
    int nearestEditorNode(int x,int y,double radiusPixels=16)const{if(track.points.empty())return -1;Transform t=editorTransform();V2 w=t.world(x,y);auto it=std::min_element(track.points.begin(),track.points.end(),[&](V2 a,V2 b){return length(a-w)<length(b-w);});int i=int(it-track.points.begin());return length(*it-w)*t.scale<=radiusPixels?i:-1;}
    int nearestEditorPitNode(int x,int y,double radiusPixels=16)const{if(track.pitlanePoints.empty())return -1;Transform t=editorTransform();V2 w=t.world(x,y);auto it=std::min_element(track.pitlanePoints.begin(),track.pitlanePoints.end(),[&](V2 a,V2 b){return length(a-w)<length(b-w);});int i=int(it-track.pitlanePoints.begin());return length(*it-w)*t.scale<=radiusPixels?i:-1;}
    bool pitlaneReady()const{if(!track.features.contains("pit_entry")||!track.features.contains("pit_exit")||!track.features["pit_entry"].is_number_integer()||!track.features["pit_exit"].is_number_integer())return false;int entry=track.features["pit_entry"].get<int>(),exit=track.features["pit_exit"].get<int>();return entry!=exit&&entry>=0&&exit>=0&&entry<int(track.points.size())&&exit<int(track.points.size());}
    void shiftFeatureArray(const char* key,int removed){if(!track.features.contains(key)||!track.features[key].is_array())return;json shifted=json::array();for(const auto&v:track.features[key])if(v.is_number_integer()){int index=v.get<int>();if(index!=removed)shifted.push_back(index>removed?index-1:index);}track.features[key]=std::move(shifted);}
    void deleteRouteNode(int node){if(node<0||node>=int(track.points.size()))return;track.points.erase(track.points.begin()+node);track.widths.erase(track.widths.begin()+node);track.grassWidths.erase(track.grassWidths.begin()+node);std::set<int> shifted;for(int index:track.kerbs)if(index!=node)shifted.insert(index>node?index-1:index);track.kerbs=std::move(shifted);shiftFeatureArray("sectors",node);for(const char* key:{"start_finish","pit_entry","pit_exit","drs_detection","drs_entry","drs_exit"})if(track.features.contains(key)&&track.features[key].is_number_integer()){int index=track.features[key].get<int>();if(index==node)track.features[key]=track.points.empty()?json(nullptr):json(std::min(node,int(track.points.size())-1));else if(index>node)track.features[key]=index-1;}if(!pitlaneReady()){track.pitlanePoints.clear();track.pitlaneWidths.clear();track.pitlaneGrassWidths.clear();track.features["pit_boxes"]=json::array();track.features["pit_start_finish"]=nullptr;}track.rebuild();selectedEditorNode=-1;}
    void deletePitNode(int node){if(node<0||node>=int(track.pitlanePoints.size()))return;track.pitlanePoints.erase(track.pitlanePoints.begin()+node);track.pitlaneWidths.erase(track.pitlaneWidths.begin()+node);track.pitlaneGrassWidths.erase(track.pitlaneGrassWidths.begin()+node);shiftFeatureArray("pit_boxes",node);if(track.features.contains("pit_start_finish")&&track.features["pit_start_finish"].is_number_integer()){int index=track.features["pit_start_finish"].get<int>();if(track.pitlanePoints.empty())track.features["pit_start_finish"]=nullptr;else if(index==node)track.features["pit_start_finish"]=std::min(node,int(track.pitlanePoints.size())-1);else if(index>node)track.features["pit_start_finish"]=index-1;}selectedEditorNode=-1;}
    void clearTrackEditor(){track.points.clear();track.widths.clear();track.grassWidths.clear();track.pitlanePoints.clear();track.pitlaneWidths.clear();track.pitlaneGrassWidths.clear();track.kerbs.clear();track.features={{"start_finish",0},{"pit_start_finish",nullptr},{"sectors",json::array()},{"pit_entry",nullptr},{"pit_exit",nullptr},{"pit_boxes",json::array()},{"drs_detection",nullptr},{"drs_entry",nullptr},{"drs_exit",nullptr}};track.rebuild();selectedEditorNode=-1;}
    void mouseDown(const SDL_MouseButtonEvent& b,int x,int y){
        if(mode==Mode::TrackEditor&&inside(x,y,{30,115,1180,745})){
            lastMouseX=x;lastMouseY=y;if(b.button==SDL_BUTTON_MIDDLE){editorPanning=true;return;}int route=nearestEditorNode(x,y,22),pit=nearestEditorPitNode(x,y,22);
            if(b.button==SDL_BUTTON_RIGHT){if(pit>=0&&(route<0||length(track.pitlanePoints[size_t(pit)]-editorTransform().world(x,y))<=length(track.points[size_t(route)]-editorTransform().world(x,y))))deletePitNode(pit);else if(route>=0)deleteRouteNode(route);return;}
            if(b.button!=SDL_BUTTON_LEFT)return;Transform t=editorTransform();V2 world=t.world(x,y);
            if(editorTool=="delete"){if(pit>=0)deletePitNode(pit);else if(route>=0)deleteRouteNode(route);else notice="CLICK A ROUTE OR PIT NODE";return;}
            if(editorTool=="pitlane"){if(!pitlaneReady()){notice="PLACE PIT IN AND PIT OUT FIRST";return;}if(pit>=0){selectedEditorNode=pit;selectedEditorPit=true;draggedPitNode=pit;}else{track.pitlanePoints.push_back(world);track.pitlaneWidths.push_back(6.0);track.pitlaneGrassWidths.push_back(16.0);selectedEditorNode=int(track.pitlanePoints.size())-1;selectedEditorPit=true;draggedPitNode=selectedEditorNode;}return;}
            if(editorTool=="pit_box"||editorTool=="pit_finish"){if(pit<0){notice="SELECT A PIT ROAD NODE";return;}selectedEditorNode=pit;selectedEditorPit=true;if(editorTool=="pit_box"){auto&boxes=track.features["pit_boxes"];if(!boxes.is_array())boxes=json::array();auto found=std::find(boxes.begin(),boxes.end(),pit);if(found==boxes.end())boxes.push_back(pit);else boxes.erase(found);}else track.features["pit_start_finish"]=pit;return;}
            if(editorTool=="route"){if(route>=0){selectedEditorNode=route;selectedEditorPit=false;draggedNode=route;}else{track.points.push_back(world);track.widths.push_back(track.widths.empty()?9.0:track.widths.back());track.grassWidths.push_back(track.grassWidths.empty()?24.0:track.grassWidths.back());track.rebuild();selectedEditorNode=int(track.points.size())-1;selectedEditorPit=false;draggedNode=selectedEditorNode;}return;}
            if(route<0){notice="CLICK A ROUTE NODE";return;}selectedEditorNode=route;selectedEditorPit=false;if(editorTool=="kerb"){if(track.kerbs.count(route))track.kerbs.erase(route);else track.kerbs.insert(route);}else if(editorTool=="sector"){auto&sectors=track.features["sectors"];if(!sectors.is_array())sectors=json::array();auto found=std::find(sectors.begin(),sectors.end(),route);if(found==sectors.end())sectors.push_back(route);else sectors.erase(found);}else if(editorTool=="start"){track.features["start_finish"]=route;track.rebuild();}else if(editorTool=="pit_entry"||editorTool=="pit_exit"||editorTool=="drs_detection"||editorTool=="drs_entry"||editorTool=="drs_exit")track.features[editorTool]=route;return;
        }
        if(mode==Mode::Algorithm&&b.button==SDL_BUTTON_LEFT){cursor=editorIndexAt(x,y);anchor=cursor;selecting=true;return;}
        if(mode==Mode::ReplaySetup&&b.button==SDL_BUTTON_LEFT){if(y>690)startReplay();else if(y>190&&y<600&&!replayFiles.empty()){replayIndex=size_t((y-190)/42)%replayFiles.size();}return;}
        if((mode==Mode::Training||mode==Mode::Race||mode==Mode::Hotlap)&&b.button==SDL_BUTTON_LEFT&&x<340&&y>130){int row=(y-167)/31;std::vector<int> order(cars.size());for(size_t i=0;i<cars.size();++i)order[i]=int(i);std::sort(order.begin(),order.end(),[&](int a,int c){return cars[a].lap==cars[c].lap?cars[a].s>cars[c].s:cars[a].lap>cars[c].lap;});if(row>=0&&row<int(order.size()))focus=order[size_t(row)];}
        if(mode==Mode::Replay&&b.button==SDL_BUTTON_LEFT&&x<340&&y>130&&!replay.empty()){int row=(y-167)/31;auto it=std::lower_bound(replay.begin(),replay.end(),replayTime,[](const ReplayFrame&f,double v){return f.time<v;});const auto&frame=it==replay.end()?replay.back():*it;if(row>=0&&row<int(frame.cars.size()))focus=row;}
    }
    void mouseMotion(int x,int y){if(mode!=Mode::TrackEditor)return;if(editorPanning){editorPan.x+=x-lastMouseX;editorPan.y+=y-lastMouseY;lastMouseX=x;lastMouseY=y;return;}if(draggedNode>=0&&draggedNode<int(track.points.size())){track.points[size_t(draggedNode)]=editorTransform().world(x,y);track.rebuild();}else if(draggedPitNode>=0&&draggedPitNode<int(track.pitlanePoints.size()))track.pitlanePoints[size_t(draggedPitNode)]=editorTransform().world(x,y);}
    void wheel(const SDL_MouseWheelEvent& w,int x,int y){
        if(mode==Mode::TrackEditor&&inside(x,y,{30,115,1180,745})){
            int node=draggedNode>=0?draggedNode:nearestEditorNode(x,y),pit=draggedPitNode>=0?draggedPitNode:nearestEditorPitNode(x,y);
            if((SDL_GetMouseState(nullptr,nullptr)&SDL_BUTTON_LMASK)&&pit>=0){track.pitlaneWidths[size_t(pit)]=clampv(track.pitlaneWidths[size_t(pit)]+w.y*.5,4.0,18.0);track.pitlaneGrassWidths[size_t(pit)]=std::max(track.pitlaneGrassWidths[size_t(pit)],track.pitlaneWidths[size_t(pit)]+2.0);selectedEditorNode=pit;selectedEditorPit=true;return;}
            if((SDL_GetMouseState(nullptr,nullptr)&SDL_BUTTON_LMASK)&&node>=0){track.widths[size_t(node)]=clampv(track.widths[size_t(node)]+w.y*.5,6.0,44.0);track.grassWidths[size_t(node)]=std::max(track.grassWidths[size_t(node)],track.widths[size_t(node)]+4.0);selectedEditorNode=node;selectedEditorPit=false;notice="NODE WIDTH  "+std::to_string(track.widths[size_t(node)]).substr(0,4)+" M";return;}
            double factor=w.y>0?1.12:1.0/1.12;double old=editorZoom;editorZoom=clampv(editorZoom*factor,.25,8.0);factor=editorZoom/old;double cx=30+1180*.5,cy=115+745*.5;editorPan.x=(x-cx)-(x-cx-editorPan.x)*factor;editorPan.y=(y-cy)-(y-cy-editorPan.y)*factor;return;
        }
        if(mode==Mode::Training||mode==Mode::Race||mode==Mode::Hotlap||mode==Mode::Replay)cameraZoom=clampv(cameraZoom*(w.y>0?1.15:1.0/1.15),MIN_CAMERA_ZOOM,MAX_CAMERA_ZOOM);
    }
    void insert(const std::string& s){snapshot();eraseSelection();editorSource.insert(cursor,s);cursor+=s.size();anchor=cursor;}
    void snapshot(){undo.push_back(editorSource);if(undo.size()>100)undo.erase(undo.begin());redo.clear();}
    std::pair<size_t,size_t> selection()const{return {std::min(cursor,anchor),std::max(cursor,anchor)};}
    void eraseSelection(){auto[a,b]=selection();if(a!=b){editorSource.erase(a,b-a);cursor=anchor=a;}}
    void editorKey(SDL_Keycode key,bool cmd,bool shift){
        if(key==SDLK_ESCAPE){mode=Mode::Menu;return;} auto[a,b]=selection();
        if(cmd&&key==SDLK_a){anchor=0;cursor=editorSource.size();return;}
        if(cmd&&key==SDLK_c){if(a!=b)SDL_SetClipboardText(editorSource.substr(a,b-a).c_str());return;}
        if(cmd&&key==SDLK_x){if(a!=b){SDL_SetClipboardText(editorSource.substr(a,b-a).c_str());snapshot();eraseSelection();}return;}
        if(cmd&&key==SDLK_v){char* clip=SDL_GetClipboardText();if(clip){std::string value=clip;SDL_free(clip);value.erase(std::remove(value.begin(),value.end(),'\r'),value.end());insert(value);}return;}
        if(cmd&&key==SDLK_s){saveAlgorithm();return;}
        if(cmd&&(key==SDLK_z||key==SDLK_y)){auto&from=(key==SDLK_z&&!shift)?undo:redo;auto&to=(key==SDLK_z&&!shift)?redo:undo;if(!from.empty()){to.push_back(editorSource);editorSource=from.back();from.pop_back();cursor=anchor=std::min(cursor,editorSource.size());}return;}
        if(key==SDLK_BACKSPACE){if(a!=b){snapshot();eraseSelection();}else if(cursor){snapshot();editorSource.erase(--cursor,1);anchor=cursor;}return;}
        if(key==SDLK_DELETE){if(a!=b){snapshot();eraseSelection();}else if(cursor<editorSource.size()){snapshot();editorSource.erase(cursor,1);}return;}
        if(key==SDLK_RETURN){insert("\n");return;}if(key==SDLK_TAB){insert("    ");return;}
        if(key==SDLK_LEFT){if(!shift)anchor=cursor;if(cursor)--cursor;if(!shift)anchor=cursor;return;}
        if(key==SDLK_RIGHT){if(!shift)anchor=cursor;if(cursor<editorSource.size())++cursor;if(!shift)anchor=cursor;return;}
        if(key==SDLK_HOME){cursor=editorSource.rfind('\n',cursor?cursor-1:0);cursor=cursor==std::string::npos?0:cursor+1;if(!shift)anchor=cursor;return;}
        if(key==SDLK_END){cursor=editorSource.find('\n',cursor);if(cursor==std::string::npos)cursor=editorSource.size();if(!shift)anchor=cursor;return;}
    }
    size_t editorIndexAt(int x,int y)const{
        int line=std::max(0,(y-145)/18)+editorScroll,col=std::max(0,(x-86)/12);size_t index=0;for(int i=0;i<line&&index<editorSource.size();++i){auto n=editorSource.find('\n',index);index=n==std::string::npos?editorSource.size():n+1;}auto n=editorSource.find('\n',index);size_t end=n==std::string::npos?editorSource.size():n;return std::min(index+size_t(col),end);
    }
    void loadAlgorithm(){auto list=filesFor("algorithms",".fai");fs::path chosen;if(!list.empty())for(auto&p:list)if(p.filename()==(hybrid?"hybrid_controller.fai":"ice_controller.fai"))chosen=p;if(chosen.empty()&&!list.empty())chosen=list.front();editorSource=readFile(chosen);cursor=anchor=editorSource.size();undo.clear();redo.clear();editorScroll=0;}
    void saveAlgorithm(){fs::path p=localData()/"algorithms"/(hybrid?"native_hybrid.fai":"native_ice.fai");if(writeFile(p,editorSource))notice="SAVED "+p.filename().string();else notice="SAVE FAILED";refreshFiles();}
    void saveTrack(){if(track.points.size()<3){notice="ADD AT LEAST 3 NODES";return;}if(!track.pitlanePoints.empty()&&!pitlaneReady()){notice="PIT ROAD NEEDS DIFFERENT PIT IN / OUT";return;}if(track.name.empty())track.name="Native Circuit";std::string filename;for(char raw:track.name){unsigned char c=static_cast<unsigned char>(raw);if(std::isalnum(c))filename.push_back(char(std::tolower(c)));else if(filename.empty()||filename.back()!='_')filename.push_back('_');}while(!filename.empty()&&filename.back()=='_')filename.pop_back();if(filename.empty())filename="native_circuit";fs::path p=localData()/"tracks"/(filename+".json");writeFile(p,track.toJson().dump(2));notice="SAVED "+p.filename().string();refreshFiles();}
    Car configuredCar(int i,bool stagger){
        static const std::array<const char*,20> names={"NOVA","APEX","VOLT","ZENITH","ORBIT","PULSE","COMET","ECHO","BLAZE","KITE","ONYX","RAPTOR","SOLAR","DRIFT","TITAN","FLUX","VEGA","STORM","RUNE","FROST"};
        Car c;c.name=names[size_t(i)%names.size()];if(i>=int(names.size()))c.name+=" "+std::to_string(i+1);c.col=palette[size_t(i)%palette.size()];c.brain=i?chosenBrain.mutate(rng,.08):chosenBrain;c.hybrid=hybrid;c.battery=hybrid?100:0;
        // Mirrors the real standing-start grid (paired rows, side by side) rather than a single-file zigzag, so racecraft training actually rehearses the traffic shape cars meet at a real race start.
        const int row=i/2,column=i%2;const double targetDistance=row*(CAR_LENGTH_M+8.0)+column*4.0;c.s=stagger?std::fmod(track.lengthM-targetDistance+track.lengthM,track.lengthM):0;double halfWidth=clampv(track.widthAt(0)*.20,CAR_WIDTH_M*.5+.55,std::max(CAR_WIDTH_M*.5+.55,track.widthAt(0)*.5-CAR_WIDTH_M*.5-.75));c.lateral=stagger?(column?-halfWidth:halfWidth):0;auto [base,tangent]=track.at(c.s);const double coordinateMetres=track.geometryLength/std::max(1.0,track.lengthM);c.position=base+normal(tangent)*(c.lateral*coordinateMetres);c.angle=angleOf(tangent);c.previousProgress=c.s;c.raceDistance=stagger?-targetDistance:0;c.poseReady=true;return c;
    }
    void resetCars(int count,bool stagger){cars.clear();cars.reserve(size_t(count));for(int i=0;i<count;++i)cars.push_back(configuredCar(i,stagger));focus=0;simTime=0;paused=false;}
    void changePopulation(int delta){
        if(mode!=Mode::Training)return;const int target=clampv(population+delta,1,50);if(target==population)return;
        if(target>population){cars.reserve(size_t(target));for(int i=population;i<target;++i)cars.push_back(configuredCar(i,racecraft));}
        else cars.resize(size_t(target));population=target;focus=clampv(focus,0,std::max(0,target-1));notice="TRAINING FIELD  "+std::to_string(population)+" AGENTS";
    }
    void startTraining(){chosenBrain.setSource(editorSource);if(!chosenBrain.program.valid()){notice="ALGORITHM ERROR  "+chosenBrain.program.error();return;}population=clampv(population,1,50);resetCars(population,racecraft);generation=1;bestFitness=-1;champion=chosenBrain;cameraZoom=DEFAULT_CAMERA_ZOOM;mode=Mode::Training;}
    void evolve(){if(cars.empty())return;auto best=std::max_element(cars.begin(),cars.end(),[](const Car&a,const Car&b){return a.fitness<b.fitness;});if(best->fitness>bestFitness){bestFitness=best->fitness;champion=best->brain;notice="NEW ALL-TIME CHAMPION";}chosenBrain=champion;++generation;resetCars(population,racecraft);}
    void saveChampion(){json d;d["version"]=3;d["name"]="Native Champion Gen "+std::to_string(generation);d["weights"]=champion.source.empty()?json(champion.weights):json(nullptr);d["algorithm"]=champion.config;d["source"]=champion.source;d["parameters"]=champion.p;d["fitness"]=bestFitness;fs::path p=localData()/"brains"/("native_champion_gen_"+std::to_string(generation)+".json");writeFile(p,d.dump(2));notice="CHAMPION SAVED";refreshFiles();}
    void triggerRedFlag(){
        if(redFlagActive||mode!=Mode::Race||cars.empty())return;
        redFlagActive=true;redFlagSuspended=false;flagState="RED FLAG";
        redFlagSnapshotOrder=raceOrder();
        std::stable_sort(redFlagSnapshotOrder.begin(),redFlagSnapshotOrder.end(),[&](int a,int b){
            return cars[size_t(a)].alive > cars[size_t(b)].alive;
        });
        redFlagLap=1;
        for(const Car&c:cars)if(c.alive)redFlagLap=std::max(redFlagLap,c.lap);
        for(Car&c:cars){if(c.alive&&!c.finished){c.pitRequested=true;c.redFlagPitStopped=false;}}
        notice="RED FLAG DEPLOYED - 60% SPEED LIMIT - ALL CARS TO PITLANE";
    }
    void resumeRaceFromRedFlag(){
        redFlagActive=false;redFlagSuspended=false;flagState="START SEQUENCE";countdown=5.0;
        size_t gridSlot=0;
        for(int carIndex:redFlagSnapshotOrder){
            if(carIndex<0||size_t(carIndex)>=cars.size())continue;
            Car&c=cars[size_t(carIndex)];
            if(!c.alive||c.finished)continue;
            const int row=int(gridSlot)/2,column=int(gridSlot)%2;
            const double targetDistance=row*(CAR_LENGTH_M+8.0)+column*4.0;
            c.s=std::fmod(track.lengthM-targetDistance+track.lengthM,track.lengthM);
            double halfWidth=clampv(track.widthAt(0)*.20,CAR_WIDTH_M*.5+.55,std::max(CAR_WIDTH_M*.5+.55,track.widthAt(0)*.5-CAR_WIDTH_M*.5-.75));
            c.lateral=column?-halfWidth:halfWidth;
            auto[base,tangent]=track.at(c.s);
            double scale=track.geometryLength/std::max(1.0,track.lengthM);
            c.position=base+normal(tangent)*(c.lateral*scale);
            c.angle=angleOf(tangent);
            c.velocity={};c.speed=0;
            c.previousProgress=c.s;
            c.lap=redFlagLap;
            c.lapStart=simTime;
            c.pitRequested=false;
            c.inPitlane=false;
            c.pitTimer=0;
            c.redFlagPitStopped=false;
            c.pitEntryCommitted=false;
            c.pitExitStraightFrames=0;
            c.pitExitGuidanceFrames=0;
            c.battery=hybrid?100:0;
            c.health=100.0;c.wear=0.0;c.dirty=0.0;c.puncture=false;
            c.tireSlip=0.0;c.understeer=0.0;c.oversteer=0.0;
            c.stagnantFrames=0;c.lowSpeedSeconds=0;
            c.racePosition=int(gridSlot)+1;
            ++gridSlot;
        }
        notice="RACE RESTARTED - STANDING RESTART";
    }
    void startRace(){
        hybrid=raceHybrid;rainLevel=raceWeather==1?1.0:0.0;lastWeatherLap=-1;
        redFlagActive=false;redFlagSuspended=false;redFlagSnapshotOrder.clear();selectedRedFlagEntry=0;
        std::uniform_real_distribution<double>forecast(.25,.75);weatherForecast=raceWeather==2?forecast(rng):rainLevel;flagState="START SEQUENCE";flagTimer=0;
        resetCars(raceCars,true);ensureRaceEntries();
        for(size_t i=0;i<cars.size();++i){
            RaceEntry&entry=raceEntries[i];Car&c=cars[i];c.name=entry.name.empty()?"UNNAMED":entry.name;c.col=palette[size_t(entry.colorIndex)%palette.size()];c.fuel=entry.fuel;c.tyreCompound=entry.tyre;c.startingPosition=int(i)+1;c.racePosition=int(i)+1;c.fieldSize=raceCars;
            if(entry.brainIndex==-2)c.brain=Brain{};else if(entry.brainIndex==-1)c.brain=chosenBrain;else if(entry.brainIndex>=0&&size_t(entry.brainIndex)<brainFiles.size())c.brain=Brain::load(brainFiles[size_t(entry.brainIndex)]);
            const int row=int(i)/2,column=int(i)%2;const double targetDistance=row*(CAR_LENGTH_M+8.0)+column*4.0;c.s=std::fmod(track.lengthM-targetDistance+track.lengthM,track.lengthM);double halfWidth=clampv(track.widthAt(0)*.20,CAR_WIDTH_M*.5+.55,std::max(CAR_WIDTH_M*.5+.55,track.widthAt(0)*.5-CAR_WIDTH_M*.5-.75));c.lateral=column?-halfWidth:halfWidth;auto[base,tangent]=track.at(c.s);double scale=track.geometryLength/std::max(1.0,track.lengthM);c.position=base+normal(tangent)*(c.lateral*scale);c.angle=angleOf(tangent);c.previousProgress=c.s;c.raceDistance=-targetDistance;c.battery=hybrid?100:0;
        }
        capturedReplay.clear();replayCaptureAccumulator=0;replaySaved=false;countdown=5.0;cameraZoom=DEFAULT_CAMERA_ZOOM;mode=Mode::Race;
    }
    void startHotlap(){hybrid=hotlapHybrid;if(hotlapBrainIndex<0)selectEmptyBrain();else if(size_t(hotlapBrainIndex)<brainFiles.size())loadBrain(size_t(hotlapBrainIndex));resetCars(1,false);cars[0].fuel=20;cars[0].tyreCompound=0;cars[0].battery=hybrid?100:0;targetLaps=2;cameraZoom=DEFAULT_CAMERA_ZOOM;mode=Mode::Hotlap;}
    void startReplay(){if(replayFiles.empty()){notice="NO REPLAY FILES";return;}replay.clear();try{auto d=json::parse(readFile(replayFiles[replayIndex]));std::string replayTrack=d.value("track","");for(size_t i=0;i<trackFiles.size();++i){Track candidate;if(candidate.load(trackFiles[i])&&candidate.name==replayTrack){track=std::move(candidate);trackIndex=i;break;}}for(auto&f:d.at("frames")){ReplayFrame rf;rf.time=f.value("time",0.0);for(auto&c:f.at("cars")){ReplayCar rc;rc.name=c.value("name","CAR");rc.generation=c.value("generation","ICE");rc.x=c.value("x",0.0);rc.y=c.value("y",0.0);rc.angle=c.value("angle",0.0)*PI/180;rc.speed=c.value("speed_kph",0.0);rc.battery=c.value("battery",0.0);rc.fuel=c.value("fuel",0.0);rc.wear=c.value("wear",0.0);rc.rpm=c.value("rpm",0.0);rc.health=c.value("health",100.0);rc.brake=c.value("brake",0.0);rc.slipstream=c.value("slipstream",0.0);rc.lap=c.value("lap",0);rc.gear=c.value("gear",1);rc.pitstops=c.value("pitstops",0);rc.pitRequested=c.value("pit_requested",false);rc.overtake=c.value("overtake",false);rc.recharge=c.value("recharge",false);rc.drsEligible=c.value("drs_eligible",false);rc.drsActive=c.value("drs_active",false);rc.removed=c.value("removed_from_track",false);if(c.contains("color")&&c["color"].is_array()&&c["color"].size()>=3)rc.col={uint8_t(c["color"][0].get<int>()),uint8_t(c["color"][1].get<int>()),uint8_t(c["color"][2].get<int>()),255};rf.cars.push_back(rc);}replay.push_back(std::move(rf));}mode=Mode::Replay;replayTime=0;replaySpeed=1;paused=false;focus=0;cameraZoom=DEFAULT_CAMERA_ZOOM;}catch(...){notice="INVALID REPLAY";}}
    void resetTraffic(){for(Car&c:cars){c.slipstream=0;c.carAhead=0;c.carAheadDistance=1;c.carAheadSide=0;c.closingSpeed=0;c.passing=false;c.passingSide=0;c.drsGap=1e9;c.opponentData.fill(0);c.opponentPresence.fill(0);c.racePosition=1;c.fieldSize=1;c.raceAggression=0;c.gapLeader=c.gapNext=0;}}
    std::vector<int> raceOrder()const{
        std::vector<int>order(cars.size());
        for(size_t i=0;i<cars.size();++i)order[i]=int(i);
        std::sort(order.begin(),order.end(),[&](int a,int b){
            if(cars[a].alive!=cars[b].alive)return cars[a].alive>cars[b].alive;
            if(cars[a].finished!=cars[b].finished)return cars[a].finished>cars[b].finished;
            if(cars[a].finished&&cars[b].finished)return cars[a].finishTime<cars[b].finishTime;
            return cars[a].raceDistance>cars[b].raceDistance;
        });
        return order;
    }
    void updateTraffic(bool assistPassing){
        auto order=raceOrder();double leaderDistance=order.empty()?0:cars[size_t(order[0])].raceDistance;for(size_t rank=0;rank<order.size();++rank){Car&c=cars[size_t(order[rank])];c.racePosition=int(rank)+1;c.fieldSize=int(order.size());c.gapLeader=std::max(0.0,leaderDistance-c.raceDistance);c.gapNext=rank?std::max(0.0,cars[size_t(order[rank-1])].raceDistance-c.raceDistance):0;double deficit=order.size()>1?double(rank)/(order.size()-1):0;c.raceAggression=clampv(deficit*.48+clampv(c.gapLeader/(std::max(1.0,track.lengthM)*.18),0.0,1.0)*.22+clampv(c.gapNext/(std::max(1.0,track.lengthM)*.045),0.0,1.0)*.15+clampv(1-c.gapNext/65,0.0,1.0)*deficit*.10+clampv(double(c.lap)/std::max(1,targetLaps),0.0,1.0)*deficit*.05,0.0,1.0);c.drsGap=rank?c.gapNext/std::max(25.0,(cars[size_t(order[rank-1])].speed+c.speed)/7.2):1e9;}
        for(size_t i=0;i<cars.size();++i){Car&f=cars[i];f.slipstream=0;f.carAhead=0;f.carAheadDistance=1;f.carAheadSide=0;f.closingSpeed=0;f.passing=false;f.opponentData.fill(0);f.opponentPresence.fill(0);if(!f.alive||f.finished){f.passingSide=0;continue;}V2 heading{std::cos(f.angle),std::sin(f.angle)},side=normal(heading);struct Nearby{double distance;size_t index;double forward,right,vforward,vright;};std::vector<Nearby>nearby;double nearestAhead=1e9;size_t target=cars.size();for(size_t j=0;j<cars.size();++j)if(i!=j&&cars[j].alive&&!cars[j].finished){V2 relative=cars[j].position-f.position,relativeVelocity=cars[j].velocity-f.velocity;double distance=dot(relative,relative);if(distance<=1600)nearby.push_back({distance,j,clampv(dot(relative,heading)/40,-1.0,1.0),clampv(dot(relative,side)/40,-1.0,1.0),clampv(dot(relativeVelocity,heading)/1.67,-1.0,1.0),clampv(dot(relativeVelocity,side)/1.67,-1.0,1.0)});double longitudinal=dot(relative,heading),lateral=dot(relative,side);V2 otherHeading{std::cos(cars[j].angle),std::sin(cars[j].angle)};if(longitudinal>0&&longitudinal<=CAR_LENGTH_M*5&&std::abs(lateral)<=track.widthAt(f.s)*.65&&dot(heading,otherHeading)>=.82&&longitudinal<nearestAhead){nearestAhead=longitudinal;target=j;}if(longitudinal>CAR_LENGTH_M*.75&&longitudinal<=CAR_LENGTH_M*3&&std::abs(lateral)<CAR_WIDTH_M*1.3&&dot(heading,otherHeading)>.94){double strength=clampv((CAR_LENGTH_M*3-longitudinal)/(CAR_LENGTH_M*2.25),0.0,1.0)*(1-std::abs(lateral)/(CAR_WIDTH_M*1.3));f.slipstream=std::max(f.slipstream,strength);}}
            std::sort(nearby.begin(),nearby.end(),[](const Nearby&a,const Nearby&b){return a.distance<b.distance;});for(size_t n=0;n<std::min<size_t>(3,nearby.size());++n){f.opponentData[n*4]=nearby[n].forward;f.opponentData[n*4+1]=nearby[n].right;f.opponentData[n*4+2]=nearby[n].vforward;f.opponentData[n*4+3]=nearby[n].vright;f.opponentPresence[n]=1;}
            if(target<cars.size()){Car&lead=cars[target];V2 relative=lead.position-f.position;double lateral=dot(relative,side);f.carAhead=1;f.carAheadDistance=clampv(nearestAhead/(CAR_LENGTH_M*5),0.0,1.0);f.carAheadSide=clampv(lateral/std::max(1.0,track.widthAt(f.s)*.5),-1.0,1.0);f.closingSpeed=clampv((dot(f.velocity,heading)-dot(lead.velocity,heading))/1.67,-1.0,1.0);
                // Committed side with hysteresis: once passingSide is chosen, only the wider band flips it back, so a centered lead car doesn't cause frame-to-frame left/right flicker (visible as swerving).
                if(assistPassing&&nearestAhead<=CAR_LENGTH_M*4&&f.closingSpeed>.035){double leftRoom=f.rayCache[2]+f.rayCache[1]*.35,rightRoom=f.rayCache[6]+f.rayCache[7]*.35;f.passing=true;double band=CAR_WIDTH_M*(f.passingSide!=0?.9:.4);double desired=lateral>band?-1:lateral<-band?1:0;if(desired!=0)f.passingSide=desired;else if(f.passingSide==0)f.passingSide=rightRoom>=leftRoom?1:-1;}else f.passingSide=0;
            }else f.passingSide=0;
        }
    }
    void armOvertakes(){for(size_t i=0;i<cars.size();++i){Car&f=cars[i];for(auto it=f.overtakeCooldowns.begin();it!=f.overtakeCooldowns.end();){if(--it->second<=0)it=f.overtakeCooldowns.erase(it);else++it;}if(f.carCollision||!f.alive)f.overtakeCandidates.clear();V2 heading{std::cos(f.angle),std::sin(f.angle)},side=normal(heading);for(size_t j=0;j<cars.size();++j)if(i!=j&&!f.overtakeCooldowns.count(int(j))&&cars[j].alive){V2 relative=cars[j].position-f.position;double longitudinal=dot(relative,heading),lateral=dot(relative,side);V2 oh{std::cos(cars[j].angle),std::sin(cars[j].angle)};if(longitudinal>CAR_LENGTH_M*.75&&longitudinal<=CAR_LENGTH_M*5&&std::abs(lateral)<track.widthAt(f.s)*.5&&dot(heading,oh)>.78&&dot(f.velocity,heading)>dot(cars[j].velocity,heading)+.02)f.overtakeCandidates.insert(int(j));}}}
    void completeOvertakes(){for(size_t i=0;i<cars.size();++i){Car&f=cars[i];V2 heading{std::cos(f.angle),std::sin(f.angle)},side=normal(heading);for(auto it=f.overtakeCandidates.begin();it!=f.overtakeCandidates.end();){int j=*it;if(j<0||j>=int(cars.size())||f.carCollision||cars[size_t(j)].carCollision||f.outsideLimits){it=f.overtakeCandidates.erase(it);continue;}V2 relative=cars[size_t(j)].position-f.position;double longitudinal=dot(relative,heading),lateral=dot(relative,side);if(std::abs(lateral)>track.widthAt(f.s)*.72||std::abs(longitudinal)>CAR_LENGTH_M*7){it=f.overtakeCandidates.erase(it);continue;}if(longitudinal<=-CAR_LENGTH_M*.75){++f.overtakes;f.fitness+=150;f.overtakeCooldowns[j]=180;it=f.overtakeCandidates.erase(it);}else++it;}}}
    static std::optional<std::pair<V2,double>> collisionManifold(const Car&a,const Car&b){V2 delta=b.position-a.position;if(dot(delta,delta)>std::pow(CAR_LENGTH_M*1.25,2))return std::nullopt;V2 af{std::cos(a.angle),std::sin(a.angle)},bf{std::cos(b.angle),std::sin(b.angle)},as=normal(af),bs=normal(bf);std::array<V2,4>axes={af,as,bf,bs};double minimum=1e9;V2 collision=af;for(V2 axis:axes){axis=unit(axis);double ar=CAR_LENGTH_M*.5*std::abs(dot(af,axis))+CAR_WIDTH_M*.5*std::abs(dot(as,axis)),br=CAR_LENGTH_M*.5*std::abs(dot(bf,axis))+CAR_WIDTH_M*.5*std::abs(dot(bs,axis)),overlap=ar+br-std::abs(dot(delta,axis));if(overlap<=0)return std::nullopt;if(overlap<minimum){minimum=overlap;collision=axis;}}if(dot(delta,collision)<0)collision=-collision;return std::pair<V2,double>{collision,minimum};}
    int collisions(bool damage){int count=0;for(size_t i=0;i<cars.size();++i)for(size_t j=i+1;j<cars.size();++j){Car&a=cars[i];Car&b=cars[j];if(!a.alive||!b.alive||a.finished||b.finished||a.pitTimer>0||b.pitTimer>0||a.inPitlane||b.inPitlane)continue;auto manifold=collisionManifold(a,b);if(!manifold)continue;++count;auto[n,overlap]=*manifold;a.carCollision=b.carCollision=true;V2 correction=n*(overlap*.5+.08);a.position=a.position-correction;b.position=b.position+correction;double closing=dot(b.velocity-a.velocity,n),impact=std::max(0.0,-closing);V2 impulse=n*std::max(impact*.58,.025);a.velocity=a.velocity-impulse;b.velocity=b.velocity+impulse;if(damage){double amount=impact*12;a.health=std::max(0.0,a.health-amount);b.health=std::max(0.0,b.health-amount);a.alive=a.health>0;b.alive=b.health>0;}else{double penalty=4+impact*6;a.collisionPenalty+=penalty;b.collisionPenalty+=penalty;a.fitness-=penalty;b.fitness-=penalty;}}return count;}
    void captureReplayFrame(){ReplayFrame frame;frame.time=simTime;for(const Car&c:cars){ReplayCar r;r.name=c.name;r.generation=c.hybrid?"Hybrid":"ICE";r.x=c.position.x;r.y=c.position.y;r.angle=c.angle;r.speed=c.speed;r.battery=c.battery;r.fuel=c.fuel;r.wear=c.wear*100;r.rpm=c.rpm;r.health=c.health;r.throttle=c.throttle;r.brake=c.brake;r.slipstream=c.slipstream;r.lap=c.lap;r.gear=c.gear;r.pitstops=c.pitstops;r.tyre=c.tyreCompound;r.pitRequested=c.pitRequested;r.overtake=c.deploying;r.recharge=c.regen;r.drsEligible=c.drsEligible;r.drsActive=c.drsActive;r.removed=c.removed;r.col=c.col;frame.cars.push_back(std::move(r));}capturedReplay.push_back(std::move(frame));}
    void saveRaceReplay(){if(cars.empty())return;captureReplayFrame();json data;data["version"]=1;data["track"]=track.name;data["settings"]={{"cars",raceCars},{"laps",targetLaps},{"generation",hybrid?"Hybrid":"ICE"},{"weather",raceWeather}};data["frames"]=json::array();for(const auto&frame:capturedReplay){json value;value["time"]=frame.time;value["rain"]=rainLevel;value["cars"]=json::array();for(const auto&c:frame.cars)value["cars"].push_back({{"name",c.name},{"x",c.x},{"y",c.y},{"angle",c.angle*180/PI},{"color",{c.col.r,c.col.g,c.col.b}},{"speed_kph",c.speed},{"battery",c.battery},{"fuel",c.fuel},{"wear",c.wear},{"rpm",c.rpm},{"health",c.health},{"brake",c.brake},{"slipstream",c.slipstream},{"lap",c.lap},{"gear",c.gear},{"pitstops",c.pitstops},{"tyre",c.tyre},{"generation",c.generation},{"pit_requested",c.pitRequested},{"overtake",c.overtake},{"recharge",c.recharge},{"drs_eligible",c.drsEligible},{"drs_active",c.drsActive},{"removed_from_track",c.removed}});data["frames"].push_back(std::move(value));}auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();fs::path path=localData()/"replays"/("race_"+std::to_string(stamp)+".json");if(writeFile(path,data.dump(2))){notice="REPLAY SAVED  "+path.filename().string();replaySaved=true;refreshFiles();}}
    void update(double dt){
        if(paused)return;
        if(mode==Mode::Replay){if(!replay.empty())replayTime=clampv(replayTime+dt*replaySpeed,0.0,replay.back().time);return;}
        if(mode!=Mode::Training&&mode!=Mode::Race&&mode!=Mode::Hotlap)return;
        if(mode==Mode::Race&&countdown>0){countdown=std::max(0.0,countdown-dt);if(countdown<=0)flagState="GREEN";return;}
        simTime+=dt;
        if(mode==Mode::Race&&flagTimer>0){flagTimer=std::max(0.0,flagTimer-dt);if(flagTimer<=0)flagState="GREEN";}
        if(mode==Mode::Race&&raceWeather==2){int leaderLap=0;for(const Car&c:cars)leaderLap=std::max(leaderLap,c.lap);if(leaderLap>0&&leaderLap%2==0&&leaderLap!=lastWeatherLap){lastWeatherLap=leaderLap;std::uniform_real_distribution<double>chance(0,1),forecast(.15,.85);rainLevel=chance(rng)<=weatherForecast?1:0;weatherForecast=forecast(rng);}}
        const bool traffic=mode==Mode::Race||racecraft;if(traffic)updateTraffic(mode==Mode::Race);else resetTraffic();if(traffic)armOvertakes();
        for(Car&car:cars){if(car.finished){if(mode==Mode::Race)car.advanceAfterFinish(track,dt);continue;}car.update(track,dt,simTime,traffic,car.slipstream,mode==Mode::Race?rainLevel:0.0,mode==Mode::Race,&rng,mode==Mode::Race&&redFlagActive);if(mode==Mode::Race&&!car.finished){double cap=redFlagActive?3.6:0.0;if(cap>0&&length(car.velocity)>cap)car.velocity=unit(car.velocity)*cap;}}
        if(mode==Mode::Race&&redFlagActive&&!redFlagSuspended){
            bool allStopped=std::all_of(cars.begin(),cars.end(),[](const Car&c){return !c.alive||c.finished||c.redFlagPitStopped;});
            if(allStopped&&!cars.empty()){
                redFlagSuspended=true;
                notice="RACE SUSPENDED (RED FLAG) - STRATEGY & REPAIRS MENU OPEN";
                for(Car&c:cars){
                    if(c.alive&&!c.finished){
                        c.health=100.0;c.wear=0.0;c.dirty=0.0;c.puncture=false;
                        c.tireSlip=0.0;c.understeer=0.0;c.oversteer=0.0;
                        if(hybrid)c.battery=100.0;
                    }
                }
            }
        }
        if(traffic){collisions(mode==Mode::Race);completeOvertakes();}
        if(mode==Mode::Training&&simTime>60)evolve();
        if(mode==Mode::Hotlap&&!cars.empty()&&cars[0].lap>=2){cars[0].finished=true;cars[0].finishTime=simTime;paused=true;notice="HOTLAP  "+std::to_string(simTime).substr(0,6)+" S";}
        if(mode==Mode::Race&&!redFlagSuspended){
            replayCaptureAccumulator+=dt;if(replayCaptureAccumulator>=.1){replayCaptureAccumulator=0;captureReplayFrame();}
            if(!redFlagActive) for(Car&c:cars)if(!c.finished&&c.lap>=targetLaps){c.finished=true;c.finishTime=simTime;c.throttle=0;c.brake=.1;}
            for(Car&c:cars)if(!c.finished&&c.alive){
                if(c.inPitlane||c.pitTimer>0||c.redFlagPitStopped||c.pitRequested){
                    c.lowSpeedSeconds=0;
                } else {
                    if(c.speed<=10)c.lowSpeedSeconds+=dt;
                    else c.lowSpeedSeconds=0;
                    if(c.lowSpeedSeconds>=20){
                        c.alive=false;c.removed=true;c.velocity={};
                    }
                }
            }
            bool classified=std::all_of(cars.begin(),cars.end(),[](const Car&c){return c.finished||!c.alive;});
            if(classified&&!replaySaved){notice="RACE FINISHED";saveRaceReplay();}
            bool cleared=std::all_of(cars.begin(),cars.end(),[](const Car&c){return c.removed||!c.alive;});
            if(cleared)paused=true;
        }
    }
    void draw(){
        ImGui_ImplSDLRenderer2_NewFrame();ImGui_ImplSDL2_NewFrame();ImGui::NewFrame();
        updateLogicalViewport();SDL_RenderSetLogicalSize(ren,0,0);SDL_RenderSetViewport(ren,nullptr);SDL_RenderSetScale(ren,1,1);
        color(ren,rgb(0x06161b));SDL_RenderClear(ren);
        SDL_RenderSetLogicalSize(ren,W,H);
        grid();if(mode==Mode::Menu)drawMenu();else if(mode==Mode::TrackEditor)drawTrackEditor();else if(mode==Mode::Algorithm||mode==Mode::RaceSetup||mode==Mode::HotlapSetup||mode==Mode::ReplaySetup){}else drawSimulation();
        SDL_RenderSetLogicalSize(ren,0,0);SDL_RenderSetScale(ren,1,1);SDL_RenderSetViewport(ren,nullptr);
        drawImGui();ImGui::Render();
        // SDL_RenderGeometry consumes renderer coordinates, while ImGui's SDL2
        // platform backend supplies window-point coordinates. On Retina those
        // are half the framebuffer dimensions. Apply the framebuffer density
        // exactly once here; the world canvas above has its own logical scale.
        ImVec2 density=ImGui::GetIO().DisplayFramebufferScale;
        SDL_RenderSetScale(ren,std::max(1.0f,density.x),std::max(1.0f,density.y));
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(),ren);
        SDL_RenderSetScale(ren,1,1);SDL_RenderPresent(ren);
    }
    void grid(){roadJoint(ren,{1480,120},330,rgb(0x0a2c2c,150));roadJoint(ren,{120,850},300,rgb(0x102632,135));color(ren,rgb(0x0c2a30));for(int x=0;x<W;x+=80)SDL_RenderDrawLine(ren,x,0,x,H);for(int y=0;y<H;y+=80)SDL_RenderDrawLine(ren,0,y,W,y);fill(ren,{0,0,W,68},rgb(0x101a1e));fill(ren,{0,68,W,3},rgb(0x17383b));text(ren,28,23,"FORMULA AI LAB  /  NATIVE C++",rgb(0xe8f2ef),3);if(!notice.empty())text(ren,1050,26,notice,rgb(0x46e1c1),2);}
    void drawImGui(){
        constexpr ImGuiWindowFlags fixed=ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove;
        ImVec2 display=ImGui::GetIO().DisplaySize;auto pos=[&](float x,float y){return ImVec2(display.x*x/W,display.y*y/H);};auto size=[&](float x,float y){return ImVec2(display.x*x/W,display.y*y/H);};
        if(showFps){ImGui::SetNextWindowPos({display.x-150.0f,10.0f});ImGui::SetNextWindowSize({135,46});ImGui::SetNextWindowBgAlpha(.86f);ImGui::Begin("##fps-counter",nullptr,fixed|ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoNav);ImGui::TextColored(currentFps>=55?ImVec4(.27f,.88f,.76f,1):currentFps>=30?ImVec4(1,.79f,.29f,1):ImVec4(1,.33f,.39f,1),"%.1f FPS",currentFps);ImGui::End();}
        if(mode==Mode::Menu){
            ImGui::SetNextWindowPos(pos(45,100));ImGui::SetNextWindowSize(size(1055,760));
            ImGui::Begin("Workspaces",nullptr,fixed|ImGuiWindowFlags_NoTitleBar);
            const std::array<std::array<const char*,2>,5> items={{
                {"1  TRACK STUDIO","Create variable-width circuits and selective kerbs"},
                {"2  AI TRAINING","Choose code, track, brain, traffic and powertrain"},
                {"3  RACE WEEKEND","Run a physical 20-car race with drafting"},
                {"4  TWO-LAP HOTLAP","Time the selected brain for exactly two laps"},
                {"5  REPLAY THEATRE","Play, rewind, seek, and change replay cameras"}
            }};
            float cardHeight = display.y < 800 ? 82.0f : 104.0f;
            float availW = ImGui::GetContentRegionAvail().x;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            for(int i=0;i<5;++i){
                ImGui::PushID(i);
                ImVec2 p0 = ImGui::GetCursorScreenPos();
                ImVec2 p1 = {p0.x + availW, p0.y + cardHeight};
                bool clicked = ImGui::InvisibleButton(items[size_t(i)][0], {availW, cardHeight});
                bool hovered = ImGui::IsItemHovered();
                if(clicked) openWorkspace(i);
                
                // Draw background image or fallback box
                dl->AddRectFilled(p0, p1, IM_COL32(10, 24, 28, 255), 10.0f);
                if(menuBgTextures[size_t(i)]){
                    dl->AddImageRounded((ImTextureID)menuBgTextures[size_t(i)], p0, p1, {0,0}, {1,1}, hovered ? IM_COL32(255, 255, 255, 240) : IM_COL32(195, 210, 215, 150), 10.0f);
                    // Dark gradient overlay for text legibility
                    dl->AddRectFilledMultiColor(p0, p1, IM_COL32(6, 16, 20, 235), IM_COL32(6, 16, 20, 120), IM_COL32(6, 16, 20, 130), IM_COL32(6, 16, 20, 245));
                }
                // Border highlight
                dl->AddRect(p0, p1, hovered ? IM_COL32(69, 225, 193, 255) : IM_COL32(35, 75, 80, 180), 10.0f, 0, hovered ? 2.0f : 1.0f);
                
                // Left accent indicator bar
                dl->AddRectFilled({p0.x + 4, p0.y + 12}, {p0.x + 8, p1.y - 12}, hovered ? IM_COL32(69, 225, 193, 255) : IM_COL32(40, 100, 100, 220), 2.0f);
                
                // Texts
                dl->AddText(uiFont, 21.0f, {p0.x + 22, p0.y + (cardHeight < 90 ? 14 : 20)}, hovered ? IM_COL32(255, 255, 255, 255) : IM_COL32(230, 242, 239, 255), items[size_t(i)][0]);
                dl->AddText(uiFont, 14.5f, {p0.x + 22, p0.y + (cardHeight < 90 ? 44 : 54)}, hovered ? IM_COL32(69, 225, 193, 255) : IM_COL32(140, 175, 170, 255), items[size_t(i)][1]);
                
                // Right arrow icon
                dl->AddText(uiFont, 18.0f, {p1.x - 30, p0.y + cardHeight * 0.5f - 9}, hovered ? IM_COL32(69, 225, 193, 255) : IM_COL32(80, 110, 110, 180), "▶");
                
                ImGui::Dummy({0, 4});
                ImGui::PopID();
            }
            ImGui::TextDisabled("Keyboard: 1-5 opens a workspace. Escape quits.");
            ImGui::End();
            ImGui::SetNextWindowPos(pos(1120,118));ImGui::SetNextWindowSize(size(435,742));
            ImGui::SetNextWindowBgAlpha(.88f);
            ImGui::Begin("Session setup",nullptr,fixed);
            ImGui::TextColored({1,.79f,.29f,1},"ACTIVE CIRCUIT");
            if(ImGui::BeginCombo("##track",track.name.c_str())){for(size_t i=0;i<trackFiles.size();++i){Track preview;std::string label=trackFiles[i].stem().string();if(preview.load(trackFiles[i]))label=preview.name;bool selected=i==trackIndex;if(ImGui::Selectable(label.c_str(),selected))loadTrack(i);if(selected)ImGui::SetItemDefaultFocus();}ImGui::EndCombo();}
            ImGui::Text("%.3f km  |  %zu nodes",track.lengthM/1000.0,track.points.size());
            float previewHeight=display.y<800?72.0f:clampv(display.y*.17f,100.0f,155.0f);ImVec2 previewPos=ImGui::GetCursorScreenPos();ImVec2 previewSize{ImGui::GetContentRegionAvail().x,previewHeight};ImGui::InvisibleButton("##track-preview",previewSize);dl=ImGui::GetWindowDrawList();dl->AddRectFilled(previewPos,{previewPos.x+previewSize.x,previewPos.y+previewSize.y},IM_COL32(4,19,23,230),8);if(track.points.size()>1){double minx=track.points[0].x,maxx=minx,miny=track.points[0].y,maxy=miny;for(V2 p:track.points){minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);}double ps=std::min((previewSize.x-28)/std::max(1.0,maxx-minx),(previewSize.y-18)/std::max(1.0,maxy-miny));double ox=previewPos.x+(previewSize.x-(maxx-minx)*ps)*.5-minx*ps,oy=previewPos.y+(previewSize.y-(maxy-miny)*ps)*.5-miny*ps;for(size_t i=0;i<track.points.size();++i){V2 a=track.points[i],b=track.points[(i+1)%track.points.size()];ImVec2 ia{float(ox+a.x*ps),float(oy+a.y*ps)},ib{float(ox+b.x*ps),float(oy+b.y*ps)};dl->AddLine(ia,ib,IM_COL32(30,39,43,255),9);dl->AddLine(ia,ib,IM_COL32(94,106,111,255),4);}}ImGui::Separator();
            ImGui::TextColored({.29f,.64f,1,1},"RACE WEEKEND DEFAULTS");ImGui::TextDisabled("Race cars / target laps");ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x*.48f);ImGui::SliderInt("##race-grid",&raceCars,2,20);ImGui::SameLine();ImGui::SetNextItemWidth(-1);ImGui::SliderInt("##race-laps",&targetLaps,1,20);
            ImGui::Separator();ImGui::TextWrapped("AI Training has its own setup workspace, matching the Python version. Open it to select the algorithm, track, powertrain, racecraft mode and base brain.");
            ImGui::End();drawKeyHints();return;
        }
        if(mode==Mode::Algorithm){
            ImGui::SetNextWindowPos(pos(24,82));ImGui::SetNextWindowSize(size(1552,776));
            ImGui::Begin("AI Training Setup",nullptr,fixed);
            ImGui::TextColored({.27f,.88f,.76f,1},"EVOLUTION LAB");ImGui::SameLine();ImGui::TextDisabled("Configure the session, edit the controller, then run training.");
            if(ImGui::BeginTable("training-selectors",4,ImGuiTableFlags_SizingStretchSame)){
                ImGui::TableNextColumn();ImGui::TextDisabled("BASE BRAIN");ImGui::SetNextItemWidth(-1);if(ImGui::BeginCombo("##setup-brain",chosenBrain.name.c_str())){if(ImGui::Selectable("EMPTY BRAIN",chosenBrainEmpty))selectEmptyBrain();for(size_t i=0;i<brainFiles.size();++i){Brain b=Brain::load(brainFiles[i]);bool selected=!chosenBrainEmpty&&i==brainIndex;if(ImGui::Selectable(b.name.c_str(),selected))loadBrain(i);if(selected)ImGui::SetItemDefaultFocus();}ImGui::EndCombo();}
                ImGui::TableNextColumn();ImGui::TextDisabled("RACECRAFT / TRAFFIC");ImGui::Checkbox("Collisions + drafting",&racecraft);
                ImGui::TableNextColumn();ImGui::TextDisabled("POWERTRAIN");if(ImGui::RadioButton("ICE",!hybrid)&&hybrid){hybrid=false;loadAlgorithm();}ImGui::SameLine();if(ImGui::RadioButton("Hybrid",hybrid)&&!hybrid){hybrid=true;loadAlgorithm();}
                ImGui::TableNextColumn();ImGui::TextDisabled("TRACK");ImGui::SetNextItemWidth(-1);if(ImGui::BeginCombo("##setup-track",track.name.c_str())){for(size_t i=0;i<trackFiles.size();++i){Track preview;std::string label=trackFiles[i].stem().string();if(preview.load(trackFiles[i]))label=preview.name;bool selected=i==trackIndex;if(ImGui::Selectable(label.c_str(),selected))loadTrack(i);if(selected)ImGui::SetItemDefaultFocus();}ImGui::EndCombo();}
                ImGui::EndTable();
            }
            ImGui::Separator();const float editorHeight=std::max(280.0f,ImGui::GetContentRegionAvail().y-68.0f);
            if(ImGui::BeginTable("training-code",2,ImGuiTableFlags_Resizable|ImGuiTableFlags_BordersInnerV,{-1,editorHeight})){
                ImGui::TableSetupColumn("ALGORITHM",ImGuiTableColumnFlags_WidthStretch,2.6f);ImGui::TableSetupColumn("LANGUAGE REFERENCE",ImGuiTableColumnFlags_WidthStretch,1.0f);ImGui::TableHeadersRow();
                ImGui::TableNextColumn();ImGui::PushFont(monoFont);ImGui::InputTextMultiline("##algorithm-source",&editorSource,{-1,-1},ImGuiInputTextFlags_AllowTabInput);ImGui::PopFont();
                ImGui::TableNextColumn();ImGui::BeginChild("language-reference",{-1,-1},true);
                ImGui::TextColored({.27f,.88f,.76f,1},"SENSORS / TELEMETRY INPUTS");
                ImGui::TextDisabled("Vision Rays & Lasers (0..1 normalized):");
                ImGui::BulletText("far_left, left, forward, right, far_right");
                ImGui::BulletText("ray_left_90, ray_left_18, ray_right_18, ray_right_90");
                ImGui::TextDisabled("Vehicle Dynamics & Kinematics:");
                ImGui::BulletText("speed (0..1), speed_kph (km/h)");
                ImGui::BulletText("local_velocity_forward, local_velocity_lateral (-1..1)");
                ImGui::BulletText("angular_velocity (-1..1), traction (0..1)");
                ImGui::BulletText("tire_slip (0..1), understeer, oversteer");
                ImGui::TextDisabled("Powertrain & Drivetrain:");
                ImGui::BulletText("rpm (0..1), rpm_value (4000..13000)");
                ImGui::BulletText("gear (0..1), gear_number (1..8)");
                if(hybrid){
                    ImGui::TextColored({1,.79f,.29f,1},"Hybrid ERS & Battery:");
                    ImGui::BulletText("battery (0..1), battery_percent (0..100)");
                    ImGui::BulletText("regen (0..1), is_hybrid (1.0)");
                    ImGui::BulletText("overtake_active, recharge_active");
                } else {
                    ImGui::TextColored({1,.50f,.25f,1},"ICE Powertrain:");
                    ImGui::BulletText("is_hybrid (0.0), constant 100%% power");
                }
                ImGui::TextDisabled("Track & Navigation:");
                ImGui::BulletText("heading_error (-1 left .. +1 right)");
                ImGui::BulletText("racing_line_offset (-1 left .. +1 right)");
                ImGui::BulletText("lap, lap_progress (0..1), progress");
                ImGui::BulletText("corner_curvature_10, _20, _40, _70, _110, _160 (-1..1)");
                ImGui::BulletText("waypoint_5/10/20/40_forward / _right");
                ImGui::BulletText("apex_distance (0 near..1 none in 180m), apex_curvature (-1..1)");
                ImGui::BulletText("target_line_offset (-1..1): suggested in-out-in racing line, optional to use");
                ImGui::BulletText("drs_eligible, drs_active, drs_in_zone, drs_gap");
                ImGui::TextDisabled("Opponents & Racecraft:");
                ImGui::BulletText("opponent_1/2/3_forward, _right");
                ImGui::BulletText("opponent_1/2/3_velocity_forward, _right");
                ImGui::BulletText("opponent_1/2/3_present (0 or 1)");
                ImGui::BulletText("car_ahead, car_ahead_distance, car_ahead_side");
                ImGui::BulletText("closing_speed, passing, passing_side");
                ImGui::BulletText("race_position, field_size, position_deficit");
                ImGui::BulletText("gap_to_leader_m, gap_to_next_m");
                ImGui::BulletText("race_aggression (0..1), aggression_error (-1..1)");
                ImGui::BulletText("slipstream (0..1), previous_steering/throttle/brake");
                ImGui::TextDisabled("Strategy, Fuel, Tyres & Conditions:");
                ImGui::BulletText("tyre_wear (0..1), tyre_age (laps), health (0..1)");
                ImGui::BulletText("tyre_soft, tyre_medium, tyre_hard, tyre_wet");
                ImGui::BulletText("fuel (0..1), fuel_kg, rain (0..1), puncture");
                ImGui::BulletText("pitstops, pit_available (0 or 1)");
                ImGui::BulletText("off_track, car_collision, dirty_tyres");
                ImGui::Separator();
                ImGui::TextColored({.29f,.64f,1,1},"OUTPUT VARIABLES");
                ImGui::BulletText("steering = -1.0 (left) .. +1.0 (right)");
                ImGui::BulletText("throttle = 0.0 .. 1.0");
                ImGui::BulletText("brake = 0.0 .. 1.0");
                if(hybrid){
                    ImGui::BulletText("overtake = 0 (off) / 1 (deploy)");
                    ImGui::BulletText("recharge = 0 (off) / 1 (charge: 70%% ICE, +5.5%%/s)");
                }
                ImGui::BulletText("pit_request = 0 (no) / 1 (enter pit)");
                ImGui::BulletText("pit_tyre = 0 (S), 1 (M), 2 (H), 3 (W)");
                ImGui::Separator();
                ImGui::TextColored({1,.79f,.29f,1},"TRAINABLE PARAMETERS");
                ImGui::TextWrapped("gain = parameter(default, min, max)");
                ImGui::TextDisabled("Mutated automatically by genetic algorithm during Evolution Lab training.");
                ImGui::Separator();
                ImGui::TextColored({.85f,.45f,1,1},"BUILT-IN FUNCTIONS");
                ImGui::BulletText("clamp(val, low, high)");
                ImGui::BulletText("min(a, b, ...), max(a, b, ...)");
                ImGui::BulletText("abs(x), sign(x), sqrt(x)");
                ImGui::Separator();
                ImGui::TextColored({.75f,.75f,.75f,1},"OPERATORS & SYNTAX");
                ImGui::TextWrapped("+  -  *  /  %%  **  ==  !=  <  <=  >  >=");
                ImGui::TextWrapped("and  or  not  if <cond> else <other>");
                ImGui::TextWrapped("True  False  pass  # comments");
                ImGui::Separator();
                ImGui::TextDisabled("SHORTCUTS: Ctrl/Cmd+A / C / X / V / Z / Y / S  •  Tab");
                ImGui::EndChild();ImGui::EndTable();
            }
                if(ImGui::Button("RELOAD",{120,44}))loadAlgorithm();ImGui::SameLine();if(ImGui::Button("SAVE CODE",{150,44}))saveAlgorithm();ImGui::SameLine();if(ImGui::Button("BACK",{110,44}))mode=Mode::Menu;ImGui::SameLine();ImGui::SetCursorPosX(ImGui::GetWindowWidth()-245);if(ImGui::Button("RUN TRAINING",{220,44}))startTraining();
            ImGui::End();drawKeyHints();return;
        }
        if(mode==Mode::TrackEditor){
            ImGui::SetNextWindowPos(pos(1185,88));ImGui::SetNextWindowSize(size(395,768));ImGui::Begin("Track tools",nullptr,fixed);
            ImGui::TextColored({.27f,.88f,.76f,1},"TRACK STUDIO");ImGui::SetNextItemWidth(-1);ImGui::InputText("##track-name",&track.name);
            float topHalf=(ImGui::GetContentRegionAvail().x-6.0f)*0.5f;
            if(ImGui::Button("SAVE TRACK",{topHalf,38}))saveTrack();ImGui::SameLine();
            if(ImGui::Button("CLEAR",{-1,38}))clearTrackEditor();
            ImGui::Separator();ImGui::TextDisabled("AUTHORING TOOLS");
            const std::array<std::pair<const char*,const char*>,8> tools={{{"route","1 ROUTE"},{"kerb","2 KERB"},{"sector","3 SECTOR"},{"start","4 START"},{"pit_entry","5 PIT IN"},{"pit_exit","6 PIT OUT"},{"pitlane","7 PIT ROAD"},{"pit_box","8 PIT BOX"}}};
            for(size_t i=0;i<tools.size();++i){
                if(i%2)ImGui::SameLine();
                bool available=(std::string(tools[i].first)!="pitlane"||pitlaneReady())&&(std::string(tools[i].first)!="pit_box"||!track.pitlanePoints.empty());
                ImGui::BeginDisabled(!available);
                if(ImGui::Selectable(tools[i].second,editorTool==tools[i].first,0,{topHalf,30}))editorTool=tools[i].first;
                ImGui::EndDisabled();
            }
            if(ImGui::Selectable("0 PIT TIMING",editorTool=="pit_finish",0,{topHalf,28}))editorTool="pit_finish";ImGui::SameLine();
            if(ImGui::Selectable("9 DELETE",editorTool=="delete",0,{-1,28}))editorTool="delete";
            ImGui::TextDisabled("DRS MARKERS");
            float drsThird=(ImGui::GetContentRegionAvail().x-12.0f)/3.0f;
            for(const auto&tool:std::array<std::pair<const char*,const char*>,3>{{{"drs_detection","DET"},{"drs_entry","IN"},{"drs_exit","OUT"}}}){
                if(tool.first!=std::string("drs_detection"))ImGui::SameLine();
                if(ImGui::Selectable(tool.second,editorTool==tool.first,0,{drsThird,26}))editorTool=tool.first;
            }
            ImGui::Separator();ImGui::Text("Nodes: %zu  •  Pit: %zu  •  %.1f m",track.points.size(),track.pitlanePoints.size(),track.lengthM);
            ImGui::TextDisabled("Pit status: %s",pitlaneReady()?"READY":"PLACE IN / OUT");
            if(!track.points.empty()){
                ImGui::Separator();ImGui::TextColored({.27f,.88f,.76f,1},"ALL NODES (GLOBAL WIDTH)");
                double avgRoad=0;for(double w:track.widths)avgRoad+=w;avgRoad/=track.widths.size();
                double avgGrass=0;for(double g:track.grassWidths)avgGrass+=g;avgGrass/=track.grassWidths.size();
                float roadAll=float(avgRoad),grassAll=float(avgGrass);
                if(ImGui::SliderFloat("All Road##global-road",&roadAll,6.0f,44.0f,"%.1f m"))track.setAllWidths(roadAll);
                if(ImGui::Button("-0.5m##road-minus",{topHalf,26}))track.adjustAllWidths(-0.5);ImGui::SameLine();
                if(ImGui::Button("+0.5m##road-plus",{-1,26}))track.adjustAllWidths(0.5);
                if(ImGui::SliderFloat("All Grass##global-grass",&grassAll,16.0f,120.0f,"%.1f m"))track.setAllGrassWidths(grassAll);
                if(ImGui::Button("-2.0m##grass-minus",{topHalf,26}))track.adjustAllGrassWidths(-2.0);ImGui::SameLine();
                if(ImGui::Button("+2.0m##grass-plus",{-1,26}))track.adjustAllGrassWidths(2.0);
                if(!track.pitlanePoints.empty()){
                    ImGui::TextDisabled("Pit Road Global:");
                    if(ImGui::Button("Pit Rd -0.5m",{topHalf,24}))track.adjustAllPitWidths(-0.5);ImGui::SameLine();
                    if(ImGui::Button("Pit Rd +0.5m",{-1,24}))track.adjustAllPitWidths(0.5);
                    if(ImGui::Button("Pit Gr -1.0m",{topHalf,24}))track.adjustAllPitGrassWidths(-1.0);ImGui::SameLine();
                    if(ImGui::Button("Pit Gr +1.0m",{-1,24}))track.adjustAllPitGrassWidths(1.0);
                }
            }
            ImGui::Separator();ImGui::TextColored({1,.79f,.29f,1},"SELECTED NODE");
            if(selectedEditorNode>=0){
                ImGui::Text("Node %s%d",selectedEditorPit?"P":"",selectedEditorNode+1);
                if(selectedEditorPit&&selectedEditorNode<int(track.pitlanePoints.size())){
                    float road=float(track.pitlaneWidths[size_t(selectedEditorNode)]),grass=float(track.pitlaneGrassWidths[size_t(selectedEditorNode)]);
                    if(ImGui::SliderFloat("Road##selected-pit",&road,4,18,"%.1f m"))track.pitlaneWidths[size_t(selectedEditorNode)]=road;
                    if(ImGui::SliderFloat("Grass##selected-pit",&grass,std::max(8.0f,road+2),60,"%.1f m"))track.pitlaneGrassWidths[size_t(selectedEditorNode)]=grass;
                }else if(!selectedEditorPit&&selectedEditorNode<int(track.points.size())){
                    float road=float(track.widths[size_t(selectedEditorNode)]),grass=float(track.grassWidths[size_t(selectedEditorNode)]);
                    if(ImGui::SliderFloat("Road##selected-route",&road,6,44,"%.1f m")){
                        track.widths[size_t(selectedEditorNode)]=road;
                        track.grassWidths[size_t(selectedEditorNode)]=std::max(track.grassWidths[size_t(selectedEditorNode)],double(road+4));
                    }
                    if(ImGui::SliderFloat("Grass##selected-route",&grass,std::max(16.0f,road+4),120,"%.1f m"))track.grassWidths[size_t(selectedEditorNode)]=grass;
                }
            } else ImGui::TextDisabled("Click a node to select it.");
            ImGui::Separator();
            ImGui::TextDisabled("S save  •  C clear  •  +/- all road  •  [/] all grass");
            if(ImGui::Button("BACK TO MENU",{-1,36}))mode=Mode::Menu;
            ImGui::End();drawKeyHints();return;
        }
        if(mode==Mode::RaceSetup){
            ensureRaceEntries();raceCars=clampv(raceCars,2,20);selectedRaceEntry=clampv(selectedRaceEntry,0,raceCars-1);
            ImGui::SetNextWindowPos(pos(28,86));ImGui::SetNextWindowSize(size(760,768));ImGui::Begin("Starting grid",nullptr,fixed);
            ImGui::TextColored({.27f,.88f,.76f,1},"RACE WEEKEND");ImGui::TextDisabled("Click a driver to configure it. Use Grid Up/Down to change starting order.");
            if(ImGui::BeginTable("race-roster",2,ImGuiTableFlags_SizingStretchSame|ImGuiTableFlags_BordersInnerV|ImGuiTableFlags_ScrollY,{-1,-1})){
                const std::array<const char*,4> tyreShort={"S","M","H","W"};
                for(int i=0;i<raceCars;++i){
                    ImGui::TableNextColumn();
                    RaceEntry& rosterEntry=raceEntries[size_t(i)];
                    SDL_Color pc=palette[size_t(rosterEntry.colorIndex)%palette.size()];
                    ImGui::PushID(i);
                    ImGui::TextColored({pc.r/255.f,pc.g/255.f,pc.b/255.f,1},"●");
                    ImGui::SameLine();
                    std::string label=(i<9?"0":"")+std::to_string(i+1)+"  "+rosterEntry.name+"  ["+tyreShort[size_t(rosterEntry.tyre)]+"]";
                    if(ImGui::Selectable(label.c_str(),selectedRaceEntry==i,0,{0,32}))selectedRaceEntry=i;
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }ImGui::End();
            ImGui::SetNextWindowPos(pos(806,86));ImGui::SetNextWindowSize(size(766,768));ImGui::Begin("Pre-race setup",nullptr,fixed);
            ImGui::TextColored({.29f,.64f,1,1},"CIRCUIT");ImGui::SetNextItemWidth(-1);
            if(ImGui::BeginCombo("##race-track",track.name.c_str())){for(size_t i=0;i<trackFiles.size();++i){Track preview;std::string label=trackFiles[i].stem().string();if(preview.load(trackFiles[i]))label=preview.name;if(ImGui::Selectable(label.c_str(),i==trackIndex))loadTrack(i);}ImGui::EndCombo();}
            ImGui::Separator();ImGui::TextColored({.27f,.88f,.76f,1},"SESSION");
            ImGui::SliderInt("Cars",&raceCars,2,20);ImGui::SliderInt("Laps",&targetLaps,1,50);
            const char* weatherNames[]={"Dry","Wet","Changing"};ImGui::Combo("Weather",&raceWeather,weatherNames,3);
            ImGui::Text("Powertrain");ImGui::SameLine();if(ImGui::RadioButton("ICE##race",!raceHybrid))raceHybrid=false;ImGui::SameLine();if(ImGui::RadioButton("Hybrid##race",raceHybrid))raceHybrid=true;
            ImGui::Checkbox("Paired teams",&raceTeams);
            ImGui::Separator();RaceEntry& entry=raceEntries[size_t(selectedRaceEntry)];
            ImGui::TextColored({1,.79f,.29f,1},"CAR %02d CONFIGURATION",selectedRaceEntry+1);
            ImGui::InputText("Driver name",&entry.name);if(raceTeams)ImGui::InputText("Team name",&raceTeamNames[size_t(selectedRaceEntry/2)]);
            const char* tyres[]={"Soft","Medium","Hard","Wet"};ImGui::Combo("Starting tyre",&entry.tyre,tyres,4);
            ImGui::SliderFloat("Fuel (kg)",&entry.fuel,5,110,"%.0f kg");
            std::string colorLabel="Livery "+std::to_string(entry.colorIndex+1);
            if(ImGui::BeginCombo("Color",colorLabel.c_str())){for(size_t i=0;i<palette.size();++i){ImGui::PushID(int(i));ImGui::TextColored({palette[i].r/255.f,palette[i].g/255.f,palette[i].b/255.f,1},"●");ImGui::SameLine();if(ImGui::Selectable(("Livery "+std::to_string(i+1)).c_str(),entry.colorIndex==int(i)))entry.colorIndex=int(i);ImGui::PopID();}ImGui::EndCombo();}
            const std::string raceBrainLabel=entry.brainIndex==-2?"EMPTY BRAIN":entry.brainIndex>=0&&size_t(entry.brainIndex)<brainFiles.size()?Brain::load(brainFiles[size_t(entry.brainIndex)]).name:"SESSION BRAIN";
            if(ImGui::BeginCombo("AI brain",raceBrainLabel.c_str())){if(ImGui::Selectable("EMPTY BRAIN",entry.brainIndex==-2))entry.brainIndex=-2;if(ImGui::Selectable("SESSION BRAIN",entry.brainIndex==-1))entry.brainIndex=-1;for(size_t i=0;i<brainFiles.size();++i){Brain b=Brain::load(brainFiles[i]);if(ImGui::Selectable(b.name.c_str(),entry.brainIndex==int(i)))entry.brainIndex=int(i);}ImGui::EndCombo();}
            float gridHalf=(ImGui::GetContentRegionAvail().x-8.0f)*0.5f;
            if(ImGui::Button("GRID UP",{gridHalf,36})&&selectedRaceEntry>0){std::swap(raceEntries[size_t(selectedRaceEntry)],raceEntries[size_t(selectedRaceEntry-1)]);--selectedRaceEntry;}
            ImGui::SameLine();
            if(ImGui::Button("GRID DOWN",{-1,36})&&selectedRaceEntry<raceCars-1){std::swap(raceEntries[size_t(selectedRaceEntry)],raceEntries[size_t(selectedRaceEntry+1)]);++selectedRaceEntry;}
            ImGui::Separator();
            float botHalf=(ImGui::GetContentRegionAvail().x-8.0f)*0.5f;
            if(ImGui::Button("BACK TO MENU",{botHalf,44}))mode=Mode::Menu;
            ImGui::SameLine();
            if(ImGui::Button("START RACE",{-1,44}))startRace();
            ImGui::End();drawKeyHints();return;
        }
        if(mode==Mode::HotlapSetup){
            ImGui::SetNextWindowPos(pos(48,88));ImGui::SetNextWindowSize(size(965,766));ImGui::Begin("Circuit preview",nullptr,fixed);
            ImGui::TextColored({.27f,.88f,.76f,1},"TWO-LAP HOTLAP");ImGui::Text("One brain. One circuit. Two timed laps.");
            ImGui::TextDisabled("The clock starts from rest and stops when lap two is completed.");
            ImVec2 previewPos=ImGui::GetCursorScreenPos(),previewSize=ImGui::GetContentRegionAvail();
            ImGui::InvisibleButton("##hotlap-preview",previewSize);
            ImDrawList* dl=ImGui::GetWindowDrawList();
            dl->AddRectFilled(previewPos,{previewPos.x+previewSize.x,previewPos.y+previewSize.y},IM_COL32(8,31,21,255),12);
            if(track.points.size()>1){
                double minx=track.points[0].x,maxx=minx,miny=track.points[0].y,maxy=miny;
                for(V2 p:track.points){minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);}
                double ps=std::min((previewSize.x-70)/std::max(1.0,maxx-minx),(previewSize.y-70)/std::max(1.0,maxy-miny));
                double ox=previewPos.x+(previewSize.x-(maxx-minx)*ps)*.5-minx*ps,oy=previewPos.y+(previewSize.y-(maxy-miny)*ps)*.5-miny*ps;
                for(size_t i=0;i<track.points.size();++i){
                    V2 a=track.points[i],b=track.points[(i+1)%track.points.size()];
                    ImVec2 ia{float(ox+a.x*ps),float(oy+a.y*ps)},ib{float(ox+b.x*ps),float(oy+b.y*ps)};
                    dl->AddLine(ia,ib,IM_COL32(24,55,35,255),14);dl->AddLine(ia,ib,IM_COL32(87,98,103,255),7);
                }
            }
            ImGui::End();
            ImGui::SetNextWindowPos(pos(1030,88));ImGui::SetNextWindowSize(size(525,766));ImGui::Begin("Run configuration",nullptr,fixed);
            ImGui::TextColored({1,.79f,.29f,1},"01  CIRCUIT");ImGui::SetNextItemWidth(-1);
            if(ImGui::BeginCombo("##hotlap-track",track.name.c_str())){for(size_t i=0;i<trackFiles.size();++i){Track preview;std::string label=trackFiles[i].stem().string();if(preview.load(trackFiles[i]))label=preview.name;if(ImGui::Selectable(label.c_str(),i==trackIndex))loadTrack(i);}ImGui::EndCombo();}
            ImGui::Text("%.3f km  •  2 timed laps",track.lengthM/1000.0);
            ImGui::Separator();ImGui::TextColored({.27f,.88f,.76f,1},"02  AI BRAIN");
            std::string hotlapBrain=hotlapBrainIndex<0?"EMPTY BRAIN":Brain::load(brainFiles[size_t(hotlapBrainIndex)]).name;ImGui::SetNextItemWidth(-1);
            if(ImGui::BeginCombo("##hotlap-brain",hotlapBrain.c_str())){if(ImGui::Selectable("EMPTY BRAIN",hotlapBrainIndex<0))hotlapBrainIndex=-1;for(size_t i=0;i<brainFiles.size();++i){Brain b=Brain::load(brainFiles[i]);if(ImGui::Selectable(b.name.c_str(),int(i)==hotlapBrainIndex))hotlapBrainIndex=int(i);}ImGui::EndCombo();}
            ImGui::TextDisabled("%zu saved + empty brain",brainFiles.size());
            ImGui::Separator();ImGui::TextColored({1,.79f,.29f,1},"03  POWERTRAIN");
            if(ImGui::RadioButton("ICE##hotlap",!hotlapHybrid))hotlapHybrid=false;ImGui::SameLine();if(ImGui::RadioButton("Hybrid##hotlap",hotlapHybrid))hotlapHybrid=true;
            ImGui::Separator();
            ImGui::TextColored({.27f,.88f,.76f,1},"FIXED CONDITIONS");ImGui::TextWrapped("Soft tyres  •  20 kg fuel  •  dry circuit");
            ImGui::Separator();
            float hlHalf=(ImGui::GetContentRegionAvail().x-8.0f)*0.5f;
            if(ImGui::Button("BACK",{hlHalf,44}))mode=Mode::Menu;
            ImGui::SameLine();
            if(ImGui::Button("START RUN",{-1,44}))startHotlap();
            ImGui::End();drawKeyHints();return;
        }
        if(mode==Mode::ReplaySetup){
            ImGui::SetNextWindowPos(pos(100,90));ImGui::SetNextWindowSize(size(1400,760));ImGui::Begin("Replay Theatre",nullptr,fixed);
            ImGui::TextColored({.70f,.40f,1,1},"SAVED JSON REPLAYS");
            float listWidth=ImGui::GetContentRegionAvail().x*.65f;
            ImGui::BeginChild("replays",{listWidth,-1},true);
            for(size_t i=0;i<replayFiles.size();++i){
                bool selected=i==replayIndex;
                if(ImGui::Selectable(replayFiles[i].stem().string().c_str(),selected,0,{0,36}))replayIndex=i;
            }
            ImGui::EndChild();
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX()+ImGui::GetContentRegionAvail().x);
            ImGui::TextWrapped("J / K / L: rewind, pause, fast-forward\nLeft / Right: seek five seconds\nUp / Down: change camera");
            ImGui::PopTextWrapPos();
            ImGui::Dummy({0,20});
            if(ImGui::Button("PLAY SELECTED",{-1,48}))startReplay();
            if(ImGui::Button("BACK TO MENU",{-1,40}))mode=Mode::Menu;
            ImGui::EndGroup();
            ImGui::End();drawKeyHints();return;
        }
        // Native overlays for all moving sessions; the circuit and cars remain
        // on the SDL renderer underneath while every interactive HUD is ImGui.
        if(mode!=Mode::Replay){
            const char* sessionName=mode==Mode::Training?"AI Evolution Lab":mode==Mode::Race?"Race Control & Timing":"Two-Lap Hotlap Timing";
            ImGui::SetNextWindowPos(pos(16,88));ImGui::SetNextWindowSize(size(560,622));
            ImGui::Begin(sessionName,nullptr,fixed);
            if(mode==Mode::Training){
                ImGui::TextColored({.27f,.88f,.76f,1},"GEN %d",generation);ImGui::SameLine();
                ImGui::TextDisabled("Best Fitness: %.0f",bestFitness);
                ImGui::TextDisabled("AGENTS");ImGui::SameLine();
                if(ImGui::Button("−##agent"))changePopulation(-1);ImGui::SameLine();
                ImGui::TextColored({1,.79f,.29f,1},"%02d",population);ImGui::SameLine();
                if(ImGui::Button("+##agent"))changePopulation(1);ImGui::SameLine();
                ImGui::TextDisabled("(1–50)");
            } else if(mode==Mode::Race) {
                int leaderLap=0;for(const Car&c:cars)leaderLap=std::max(leaderLap,c.lap);
                ImGui::TextColored({1,.79f,.29f,1},"LAP %d / %d",std::min(leaderLap+1,targetLaps),targetLaps);
                ImGui::SameLine();
                // Flag state badge
                if(flagState=="RED FLAG") ImGui::TextColored({1,.20f,.25f,1},"⛔ RED FLAG");
                else if(flagState=="GREEN") ImGui::TextColored({.22f,.88f,.45f,1},"● GREEN");
                else ImGui::TextColored({1,.79f,.29f,1},"● %s",flagState.c_str());
                if(raceWeather==2){
                    ImGui::TextColored(rainLevel>.5?ImVec4(.29f,.64f,1,1):ImVec4(.8f,.85f,.8f,1),"WEATHER: %s (Forecast Wet %.0f%%)",rainLevel>.5?"WET":"DRY",weatherForecast*100);
                }
            } else if(mode==Mode::Hotlap) {
                ImGui::TextColored({1,.79f,.29f,1},"HOTLAP  •  LAP %d / 2",cars.empty()?1:std::min(cars[0].lap+1,2));
                ImGui::SameLine();
                ImGui::TextDisabled("Time: %.2f s",simTime);
            }
            ImGui::TextDisabled("%s  •  %s",track.name.c_str(),hybrid?"HYBRID":"ICE");
            ImGui::Separator();
            auto order=raceOrder();
            // Timing tower data page: mirrors the Python HUD's cyclable metric strip (LEFT/RIGHT to page through it, UP/DOWN to change driver) as an extra column alongside the always-on ones above, rather than replacing them.
            static const std::array<const char*,9> towerLabels={"INTERVAL","GAP TO LEADER","TYRE / AGE","PIT STOPS","CONDITION","BATTERY / ENERGY","FUEL / PIT CALL","SPEED / GEAR / RPM","AGGRESSION / RISK"};
            {
                std::string pill=std::string("\xe2\x80\xb9  ")+towerLabels[size_t(timingMetric)]+"  \xe2\x80\xba";
                float pillWidth=ImGui::CalcTextSize(pill.c_str()).x;
                ImGui::SetCursorPosX(std::max(4.0f,(ImGui::GetContentRegionAvail().x-pillWidth)*.5f));
                ImGui::TextColored({.42f,.85f,1,1},"%s",pill.c_str());
            }
            if(ImGui::BeginTable("timing",7,ImGuiTableFlags_RowBg|ImGuiTableFlags_ScrollY|ImGuiTableFlags_SizingStretchProp|ImGuiTableFlags_BordersInnerH,{-1,-1})){
                ImGui::TableSetupScrollFreeze(0,1);
                ImGui::TableSetupColumn("#",ImGuiTableColumnFlags_WidthFixed,24);
                ImGui::TableSetupColumn("DRIVER",ImGuiTableColumnFlags_WidthStretch,1.4f);
                ImGui::TableSetupColumn("TYRE",ImGuiTableColumnFlags_WidthFixed,58);
                ImGui::TableSetupColumn("KM/H",ImGuiTableColumnFlags_WidthFixed,48);
                ImGui::TableSetupColumn(hybrid?"MODE":"GEAR",ImGuiTableColumnFlags_WidthFixed,58);
                ImGui::TableSetupColumn("STATUS/GAP",ImGuiTableColumnFlags_WidthFixed,78);
                ImGui::TableSetupColumn("TOWER",ImGuiTableColumnFlags_WidthFixed,104);
                ImGui::TableHeadersRow();
                static const std::array<const char*,4> tyreLetters={"S","M","H","W"};
                static const std::array<ImVec4,4> tyreColors={ImVec4(1,.25f,.30f,1),ImVec4(1,.80f,.25f,1),ImVec4(.95f,.95f,.95f,1),ImVec4(.30f,.65f,1,1)};
                for(size_t rank=0;rank<order.size();++rank){
                    int index=order[rank];const Car& c=cars[size_t(index)];
                    ImGui::PushID(index);ImGui::TableNextRow(0,25);
                    ImGui::TableSetColumnIndex(0);
                    if(c.finished) ImGui::TextColored({.27f,.88f,.76f,1},"F%zu",rank+1);
                    else if(!c.alive) ImGui::TextDisabled("DNF");
                    else ImGui::Text("%zu",rank+1);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextColored({c.col.r/255.f,c.col.g/255.f,c.col.b/255.f,1},"●");
                    ImGui::SameLine();
                    if(ImGui::Selectable(c.name.c_str(),index==focus,ImGuiSelectableFlags_SpanAllColumns))focus=index;
                    ImGui::TableSetColumnIndex(2);
                    ImGui::TextColored(tyreColors[size_t(clampv(c.tyreCompound,0,3))],"[%s] %.0f%%",tyreLetters[size_t(clampv(c.tyreCompound,0,3))],c.wear*100);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%3.0f",c.speed);
                    ImGui::TableSetColumnIndex(4);
                    if(c.hybrid){
                        if(c.deploying) ImGui::TextColored({1,.79f,.29f,1},"DEPLOY");
                        else if(c.regen) ImGui::TextColored({.29f,.64f,1,1},"REGEN");
                        else if(c.drsActive) ImGui::TextColored({.27f,.88f,.76f,1},"M.O.M.");
                        else ImGui::TextDisabled("%.0f%%",c.battery);
                    } else {
                        if(c.drsActive) ImGui::TextColored({.27f,.88f,.76f,1},"DRS");
                        else ImGui::Text("G%d",c.gear);
                    }
                    ImGui::TableSetColumnIndex(5);
                    if(!c.alive) ImGui::TextColored({1,.33f,.39f,1},"DNF");
                    else if(c.finished) ImGui::TextColored({.27f,.88f,.76f,1},"FIN");
                    else if(c.inPitlane) ImGui::TextColored({1,.65f,.20f,1},"IN PIT");
                    else if(c.pitRequested) ImGui::TextColored({1,.79f,.29f,1},"PIT CALL");
                    else if(rank==0) ImGui::TextColored({.27f,.88f,.76f,1},"LEADER");
                    else if(c.gapLeader>0) ImGui::Text("+%.1fs",c.gapLeader/std::max(20.0,c.speed/3.6));
                    else ImGui::TextDisabled("—");
                    ImGui::TableSetColumnIndex(6);
                    if(!c.alive) ImGui::TextDisabled("DNF");
                    else if(c.finished) ImGui::TextColored({.27f,.88f,.76f,1},"FINISHED");
                    else switch(timingMetric){
                        case 0: if(rank==0)ImGui::TextDisabled("LEADER");else ImGui::Text("+%.2fs",c.gapNext/std::max(20.0,c.speed/3.6)); break;
                        case 1: if(rank==0)ImGui::TextDisabled("LEADER");else ImGui::Text("+%.2fs",c.gapLeader/std::max(20.0,c.speed/3.6)); break;
                        case 2: ImGui::Text("%s %dL / %.0f%%",tyreLetters[size_t(clampv(c.tyreCompound,0,3))],c.tyreLaps,c.wear*100); break;
                        case 3: ImGui::Text("%d STOP%s",c.pitstops,c.pitstops==1?"":"S"); break;
                        case 4: if(c.slipstream>.05)ImGui::TextColored({.29f,.64f,1,1},"%.0f%% DRAFT",c.health);else ImGui::Text("%.0f%%",c.health); break;
                        case 5: if(c.hybrid)ImGui::Text("%.0f%%  %s",c.battery,c.regen?"RECHARGE":c.deploying?"OVERTAKE":c.drsActive?"M.O.M.":c.drsEligible?"READY":"—");else ImGui::TextColored(c.drsActive?ImVec4(.27f,.88f,.76f,1):c.drsEligible?ImVec4(1,.79f,.29f,1):ImVec4(.6f,.65f,.63f,1),"DRS"); break;
                        case 6: ImGui::Text("%.1fkg  %s",c.fuel,c.pitRequested?"PIT":"RUN"); break;
                        case 7: ImGui::Text("%.0fkm/h G%d %.1fk",c.speed,c.gear,c.rpm/1000.0); break;
                        default: ImGui::Text("AGG %.0f%%  %s",c.raceAggression*100,c.aggressionError>.15?"OVERCOMMIT":c.aggressionError<-.15?"HESITATE":"PUSH"); break;
                    }
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
            ImGui::End();
        }
        // Session controls bar
        ImGui::SetNextWindowPos(pos(583,88));ImGui::SetNextWindowSize({display.x*1001/W,56});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,{6,4});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,{8,4});
        ImGui::Begin("Session controls",nullptr,fixed|ImGuiWindowFlags_NoTitleBar);
        int cameraCount=int(cars.size());if(mode==Mode::Replay&&!replay.empty()){auto it=std::lower_bound(replay.begin(),replay.end(),replayTime,[](const ReplayFrame&f,double v){return f.time<v;});cameraCount=int((it==replay.end()?replay.back():*it).cars.size());}
        ImGui::TextColored({.27f,.88f,.76f,1},"%s",track.name.c_str());ImGui::SameLine();
        if(ImGui::Button(paused?"RESUME":"PAUSE"))paused=!paused;ImGui::SameLine();
        if(ImGui::Button("◀ PREV")&&cameraCount)focus=(focus-1+cameraCount)%cameraCount;ImGui::SameLine();
        if(ImGui::Button("NEXT ▶")&&cameraCount)focus=(focus+1)%cameraCount;ImGui::SameLine();
        if(mode==Mode::Training){
            if(ImGui::Button("EVOLVE"))evolve();ImGui::SameLine();
            if(ImGui::Button("SAVE CODE"))saveAlgorithm();ImGui::SameLine();
            if(ImGui::Button("SAVE BEST"))saveChampion();ImGui::SameLine();
        } else if(mode==Mode::Race){
            if(ImGui::Button("RESTART"))startRace();ImGui::SameLine();
            if(ImGui::Button(replaySaved?"REPLAY SAVED":"SAVE REPLAY"))saveRaceReplay();ImGui::SameLine();
            if(flagState!="RED FLAG"){
                if(ImGui::Button("RED FLAG"))triggerRedFlag();
            } else {
                ImGui::TextColored({1,.25f,.25f,1},"⛔ RED FLAG");
            }
            ImGui::SameLine();
        } else if(mode==Mode::Hotlap){
            if(ImGui::Button("RESTART"))startHotlap();ImGui::SameLine();
        }
        if(ImGui::Button("−##zoom"))cameraZoom=std::max(MIN_CAMERA_ZOOM,cameraZoom/1.15);ImGui::SameLine();
        ImGui::Text("Zoom %.1fx",cameraZoom);ImGui::SameLine();
        if(ImGui::Button("+##zoom"))cameraZoom=std::min(MAX_CAMERA_ZOOM,cameraZoom*1.15);ImGui::SameLine();
        if(ImGui::Button("EXIT TO MENU")){mode=Mode::Menu;cars.clear();}
        ImGui::End();
        ImGui::PopStyleVar(2);
        // Focused car rich telemetry HUD
        if(!cars.empty()&&focus<int(cars.size())){
            const Car& c=cars[size_t(focus)];
            ImGui::SetNextWindowPos(pos(583,700));ImGui::SetNextWindowSize(size(1001,156));
            ImGui::Begin("Focused car telemetry",nullptr,fixed);
            if(ImGui::BeginTable("focused-dashboard",5,ImGuiTableFlags_SizingStretchSame|ImGuiTableFlags_BordersInnerV,{-1,-1})){
                // Column 1: Driver Profile
                ImGui::TableNextColumn();
                ImGui::TextColored({c.col.r/255.f,c.col.g/255.f,c.col.b/255.f,1},"●  %s",c.name.c_str());
                ImGui::Text("Pos P%d / %d  •  Grid P%d",c.racePosition,c.fieldSize,c.startingPosition);
                if(mode==Mode::Training) ImGui::Text("Lap %d  •  Fit %.0f",c.lap,c.fitness);
                else ImGui::Text("Lap %d / %d  •  Pits %d",c.lap+1,targetLaps,c.pitstops);
                if(c.gapLeader>0) ImGui::TextDisabled("Gap: +%.2f s",c.gapLeader/std::max(20.0,c.speed/3.6));
                // Column 2: Speed / RPM / Gear
                ImGui::TableNextColumn();
                ImGui::TextColored({.27f,.88f,.76f,1},"%.1f KM/H",c.speed);
                ImGui::Text("Gear %d / 8  •  %.0f RPM",c.gear,c.rpm);
                float rpmFraction=float(clampv((c.rpm-4000.0)/9000.0,0.0,1.0));
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,rpmFraction>.85f?ImVec4(1,.25f,.25f,1):rpmFraction>.65f?ImVec4(1,.79f,.29f,1):ImVec4(.27f,.88f,.76f,1));
                ImGui::ProgressBar(rpmFraction,{-1,6},"");
                ImGui::PopStyleColor();
                ImGui::TextDisabled("Angular: %+.1f°/s",c.angularVelocity);
                // Column 3: Throttle / Brake / Steer
                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,ImVec4(.22f,.88f,.45f,1));
                ImGui::ProgressBar(float(c.throttle),{-1,16},"THROTTLE");
                ImGui::PopStyleColor();
                ImGui::Dummy({0,3});
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,ImVec4(1,.30f,.35f,1));
                ImGui::ProgressBar(float(c.brake),{-1,16},"BRAKE");
                ImGui::PopStyleColor();
                ImGui::Dummy({0,3});
                ImGui::Text("Steer: %+.2f",c.steer);
                // Column 4: Powertrain & Tyres
                ImGui::TableNextColumn();
                static const std::array<const char*,4> tNames={"Soft","Medium","Hard","Wet"};
                if(c.hybrid){
                    ImVec4 batCol=c.deploying?ImVec4(1,.79f,.29f,1):c.regen?ImVec4(.29f,.64f,1,1):ImVec4(.27f,.88f,.76f,1);
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,batCol);
                    ImGui::ProgressBar(float(c.battery/100.0),{-1,16},c.deploying?"DEPLOY":c.regen?"REGEN":"BATTERY");
                    ImGui::PopStyleColor();
                } else {
                    ImGui::TextColored({1,.50f,.25f,1},"ICE V10 POWERTRAIN");
                }
                ImGui::Dummy({0,3});
                ImGui::Text("Fuel %.1f kg  •  [%s] %.1f%%",c.fuel,tNames[size_t(clampv(c.tyreCompound,0,3))],c.wear*100.0);
                if(c.puncture) ImGui::TextColored({1,.25f,.25f,1},"⚠ TYRE PUNCTURE!");
                else if(c.outsideLimits) ImGui::TextColored({1,.65f,.20f,1},"⚠ OFF TRACK");
                else if(c.carCollision) ImGui::TextColored({1,.40f,.20f,1},"⚡ CONTACT");
                else ImGui::TextDisabled("Health: %.0f%%",c.health);
                // Column 5: Vehicle Dynamics & Racecraft
                ImGui::TableNextColumn();
                ImGui::Text("Traction: %.0f%%  •  Slip: %.0f%%",c.traction*100,c.tireSlip*100);
                if(c.slipstream>.05) ImGui::TextColored({.29f,.64f,1,1},"✈ DRAFTING %.0f%%",c.slipstream*100);
                else if(c.drsActive) ImGui::TextColored({.27f,.88f,.76f,1},"⚡ DRS ACTIVE");
                else if(c.drsEligible) ImGui::TextColored({1,.79f,.29f,1},"⚡ DRS READY");
                else ImGui::TextDisabled("Aero: Normal");
                if(c.understeer>.20) ImGui::TextColored({1,.65f,.20f,1},"Understeer: %.0f%%",c.understeer*100);
                else if(c.oversteer>.20) ImGui::TextColored({1,.40f,.20f,1},"Oversteer: %.0f%%",c.oversteer*100);
                else ImGui::TextDisabled("Balanced");
                ImGui::EndTable();
            }
            ImGui::End();
        }
        // Red Flag Race Suspension & Strategy Modal
        if(mode==Mode::Race && redFlagSuspended){
            ImGui::SetNextWindowPos(pos(140,90));ImGui::SetNextWindowSize(size(1320,730));
            ImGui::Begin("RED FLAG SUSPENSION - STRATEGY & REPAIRS",nullptr,fixed);
            ImGui::TextColored({1,.25f,.25f,1},"⛔ RED FLAG SUSPENSION  •  LAP %d / %d",redFlagLap,targetLaps);
            ImGui::TextWrapped("Under FIA F1 Article 57 regulations, the race is suspended with all surviving cars safely in the pitlane. All active cars have returned to 100%% health and condition. You may change tyre compounds and adjust fuel onboard for each car before the standing restart. Retired/DNF cars cannot be edited.");
            ImGui::Separator();
            
            if(ImGui::BeginTable("redflag-roster",2,ImGuiTableFlags_Resizable|ImGuiTableFlags_BordersInnerV,{-1,display.y*520/H})){
                ImGui::TableSetupColumn("GRID ORDER & SURVIVING CARS",ImGuiTableColumnFlags_WidthStretch,1.1f);
                ImGui::TableSetupColumn("CAR STRATEGY CONFIGURATION",ImGuiTableColumnFlags_WidthStretch,1.9f);
                ImGui::TableHeadersRow();
                
                ImGui::TableNextColumn();
                static const std::array<const char*,4> tyreLetters={"S","M","H","W"};
                for(size_t rank=0;rank<redFlagSnapshotOrder.size();++rank){
                    int carIdx=redFlagSnapshotOrder[rank];
                    if(carIdx<0||size_t(carIdx)>=cars.size()) continue;
                    Car& c=cars[size_t(carIdx)];
                    ImGui::PushID(int(rank));
                    SDL_Color pc=c.col;
                    ImGui::TextColored({pc.r/255.f,pc.g/255.f,pc.b/255.f,1},"●");
                    ImGui::SameLine();
                    std::string label=(rank<9?"P0":"P")+std::to_string(rank+1)+"  "+c.name;
                    if(!c.alive) label+="  [RETIRED / DNF]";
                    else label+="  ["+std::string(tyreLetters[size_t(clampv(c.tyreCompound,0,3))])+"]  "+std::to_string(int(c.fuel))+"kg";
                    bool selected=(selectedRedFlagEntry==int(rank));
                    if(ImGui::Selectable(label.c_str(),selected,0,{0,34})) selectedRedFlagEntry=int(rank);
                    ImGui::PopID();
                }
                
                ImGui::TableNextColumn();
                if(selectedRedFlagEntry>=0&&size_t(selectedRedFlagEntry)<redFlagSnapshotOrder.size()){
                    int selectedCarIdx=redFlagSnapshotOrder[size_t(selectedRedFlagEntry)];
                    if(selectedCarIdx>=0&&size_t(selectedCarIdx)<cars.size()){
                        Car& targetCar=cars[size_t(selectedCarIdx)];
                        ImGui::TextColored({1,.79f,.29f,1},"P%02d - %s",selectedRedFlagEntry+1,targetCar.name.c_str());
                        ImGui::Separator();
                        if(!targetCar.alive){
                            ImGui::TextColored({1,.25f,.25f,1},"STATUS: RETIRED / DNF (CANNOT BE EDITED)");
                            ImGui::TextDisabled("This vehicle suffered terminal damage or retired before the red flag and cannot take the restart.");
                        } else {
                            ImGui::TextColored({.27f,.88f,.76f,1},"STATUS: 100%% REPAIRED & READY FOR RESTART");
                            ImGui::Dummy({0,10});
                            const char* tyreNames[]={"Soft (S)","Medium (M)","Hard (H)","Wet (W)"};
                            int currentTyre=targetCar.tyreCompound;
                            if(ImGui::Combo("Tire Compound",&currentTyre,tyreNames,4)){
                                targetCar.tyreCompound=currentTyre;
                                targetCar.requestedTyre=currentTyre;
                            }
                            float carFuel=float(targetCar.fuel);
                            if(ImGui::SliderFloat("Fuel Onboard (kg)",&carFuel,5.0f,110.0f,"%.0f kg")){
                                targetCar.fuel=carFuel;
                            }
                            ImGui::Dummy({0,10});
                            ImGui::Text("Vehicle Health: 100%%  •  Tire Wear: 0%%  •  Dirt: 0%%");
                            if(hybrid) ImGui::Text("Battery ERS: 100%% (Fully Charged)");
                        }
                    }
                }
                ImGui::EndTable();
            }
            
            ImGui::Separator();
            if(ImGui::Button("RESUME RACE (STANDING RESTART)",{-1,52})){
                resumeRaceFromRedFlag();
            }
            ImGui::End();
        }
        // Race Classification / Results Modal
        if(mode==Mode::Race && !cars.empty() && !redFlagSuspended){
            bool allClassified=std::all_of(cars.begin(),cars.end(),[](const Car&c){return c.finished||!c.alive;});
            if(allClassified){
                ImGui::SetNextWindowPos(pos(180,110));ImGui::SetNextWindowSize(size(1240,680));
                ImGui::Begin("Race Classification Results",nullptr,fixed);
                ImGui::TextColored({.27f,.88f,.76f,1},"RACE CLASSIFICATION  •  %s  •  %d LAPS",track.name.c_str(),targetLaps);
                ImGui::Separator();
                auto order=raceOrder();
                if(ImGui::BeginTable("final-results",7,ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerV|ImGuiTableFlags_ScrollY,{-1,display.y*460/H})){
                    ImGui::TableSetupScrollFreeze(0,1);
                    ImGui::TableSetupColumn("POS",ImGuiTableColumnFlags_WidthFixed,45);
                    ImGui::TableSetupColumn("DRIVER",ImGuiTableColumnFlags_WidthStretch,1.6f);
                    ImGui::TableSetupColumn("STATUS",ImGuiTableColumnFlags_WidthFixed,110);
                    ImGui::TableSetupColumn("TOTAL TIME",ImGuiTableColumnFlags_WidthFixed,110);
                    ImGui::TableSetupColumn("BEST LAP",ImGuiTableColumnFlags_WidthFixed,100);
                    ImGui::TableSetupColumn("PIT STOPS",ImGuiTableColumnFlags_WidthFixed,80);
                    ImGui::TableSetupColumn("GRID",ImGuiTableColumnFlags_WidthFixed,55);
                    ImGui::TableHeadersRow();
                    for(size_t i=0;i<order.size();++i){
                        const Car& c=cars[size_t(order[i])];
                        ImGui::TableNextRow(0,28);
                        ImGui::TableSetColumnIndex(0);
                        if(i==0) ImGui::TextColored({1,.85f,.20f,1},"P1 🏆");
                        else if(i==1) ImGui::TextColored({.85f,.88f,.92f,1},"P2 🥈");
                        else if(i==2) ImGui::TextColored({.80f,.50f,.20f,1},"P3 🥉");
                        else if(c.finished) ImGui::Text("P%zu",i+1);
                        else ImGui::TextColored({1,.33f,.39f,1},"DNF");
                        ImGui::TableSetColumnIndex(1);
                        ImGui::TextColored({c.col.r/255.f,c.col.g/255.f,c.col.b/255.f,1},"●  %s",c.name.c_str());
                        ImGui::TableSetColumnIndex(2);
                        if(c.finished) ImGui::TextColored({.22f,.88f,.45f,1},"FINISHED");
                        else ImGui::TextColored({1,.33f,.39f,1},"RETIRED");
                        ImGui::TableSetColumnIndex(3);
                        if(c.finishTime>0) ImGui::Text("%.3f s",c.finishTime);
                        else ImGui::TextDisabled("—");
                        ImGui::TableSetColumnIndex(4);
                        if(c.bestLap<1e8) ImGui::Text("%.3f s",c.bestLap);
                        else ImGui::TextDisabled("—");
                        ImGui::TableSetColumnIndex(5);
                        ImGui::Text("%d stop%s",c.pitstops,c.pitstops==1?"":"s");
                        ImGui::TableSetColumnIndex(6);
                        ImGui::Text("P%d",c.startingPosition);
                    }
                    ImGui::EndTable();
                }
                ImGui::Separator();
                if(ImGui::Button(replaySaved?"REPLAY SAVED TO DISK":"SAVE RACE REPLAY",{220,44}))saveRaceReplay();
                ImGui::SameLine();
                if(ImGui::Button("RESTART RACE",{180,44}))startRace();
                ImGui::SameLine();
                if(ImGui::Button("RETURN TO MENU",{180,44})){mode=Mode::Menu;cars.clear();}
                ImGui::End();
            }
        }
        // Hotlap Completion Modal
        if(mode==Mode::Hotlap && !cars.empty() && cars[0].finished){
            ImGui::SetNextWindowPos(pos(450,220));ImGui::SetNextWindowSize(size(700,420));
            ImGui::Begin("Two-Lap Hotlap Results",nullptr,fixed);
            ImGui::TextColored({.27f,.88f,.76f,1},"TWO-LAP HOTLAP COMPLETED");
            ImGui::Separator();
            ImGui::Text("Circuit: %s  (%.3f km)",track.name.c_str(),track.lengthM/1000.0);
            ImGui::Text("Driver / Brain: %s",cars[0].name.c_str());
            ImGui::Dummy({0,10});
            ImGui::TextColored({1,.79f,.29f,1},"TOTAL TIME: %.3f s",cars[0].finishTime);
            if(cars[0].bestLap<1e8) ImGui::TextColored({.22f,.88f,.45f,1},"BEST LAP:   %.3f s",cars[0].bestLap);
            ImGui::Dummy({0,20});
            if(ImGui::Button("RUN AGAIN",{180,48}))startHotlap();
            ImGui::SameLine();
            if(ImGui::Button("RETURN TO MENU",{180,48})){mode=Mode::Menu;cars.clear();}
            ImGui::End();
        }
        // Replay transport controls
        if(mode==Mode::Replay&&!replay.empty()){
            ImGui::SetNextWindowPos(pos(480,788));ImGui::SetNextWindowSize(size(740,62));
            ImGui::Begin("Replay transport",nullptr,fixed|ImGuiWindowFlags_NoTitleBar);
            if(ImGui::Button("◀◀ -2x"))replaySpeed=-2;ImGui::SameLine();
            if(ImGui::Button(paused?"▶ PLAY":"❚❚ PAUSE")){paused=!paused;replaySpeed=paused?0:1;}ImGui::SameLine();
            if(ImGui::Button("1x"))replaySpeed=1;ImGui::SameLine();
            if(ImGui::Button("2x ▶▶"))replaySpeed=2;ImGui::SameLine();
            float value=float(replayTime),maximum=float(replay.back().time);
            ImGui::SetNextItemWidth(display.x*280/W);
            if(ImGui::SliderFloat("##timeline",&value,0,maximum,"%.1f s"))replayTime=value;
            ImGui::SameLine();
            ImGui::TextDisabled("Camera: %s",cars.empty()?"":cars[size_t(clampv(focus,0,int(cars.size())-1))].name.c_str());
            ImGui::End();
        }
        drawKeyHints();
    }
    // Persistent keyboard-shortcut legend, docked along the bottom edge of every screen so controls are always visible without covering any existing panel.
    void drawKeyHints(){
        ImVec2 display=ImGui::GetIO().DisplaySize;auto pos=[&](float x,float y){return ImVec2(display.x*x/W,display.y*y/H);};auto size=[&](float x,float y){return ImVec2(display.x*x/W,display.y*y/H);};
        std::string hint;
        switch(mode){
            case Mode::Menu: hint="1-5 Open Workspace   •   ←/→ Track   •   B/N Brain   •   G Powertrain   •   Esc Quit"; break;
            case Mode::TrackEditor: hint="1-9 Tool   •   0 Pit/Finish Line   •   S Save   •   C Clear   •   -/+ Road Width   •   [ / ] Grass Width   •   Esc Menu"; break;
            case Mode::Algorithm: hint="Ctrl+S Save   •   Ctrl+Z / Ctrl+Y Undo / Redo   •   Ctrl+A / C / X / V Select / Copy / Cut / Paste   •   Esc Menu"; break;
            case Mode::RaceSetup: hint="Click a Driver to Configure   •   GRID UP/DOWN Reorder   •   Enter Start Race   •   Esc Menu"; break;
            case Mode::HotlapSetup: hint="←/→ or ↑/↓ Choose Brain   •   Enter / Space Start Hotlap   •   Esc Menu"; break;
            case Mode::ReplaySetup: hint="J / K / L Rewind / Pause / Fast-Forward   •   ←/→ Seek 5s   •   ↑/↓ Camera   •   Esc Menu"; break;
            case Mode::Training: hint="Space Pause   •   R Evolve   •   S Save Best   •   -/+ Agents   •   [ / ] Zoom   •   ←/→ Tower Data   •   ↑/↓ Driver   •   Esc Menu"; break;
            case Mode::Race: hint="Space Pause   •   R Red Flag   •   S Save Replay   •   W Toggle Rain   •   [ / ] Zoom   •   ←/→ Tower Data   •   ↑/↓ Driver   •   Esc Menu"; break;
            case Mode::Hotlap: hint="Space Pause   •   R Restart   •   [ / ] Zoom   •   ↑/↓ Driver   •   Esc Menu"; break;
            case Mode::Replay: hint="J / K / L Rewind / Pause / Fast-Forward   •   ←/→ Seek 5s   •   ↑/↓ Camera   •   Esc Menu"; break;
        }
        ImGui::SetNextWindowPos(pos(8,864));ImGui::SetNextWindowSize(size(1584,32));ImGui::SetNextWindowBgAlpha(.88f);
        ImGui::Begin("##key-hints",nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoDecoration);
        float textWidth=ImGui::CalcTextSize(hint.c_str()).x;
        ImGui::SetCursorPosX(std::max(8.0f,(ImGui::GetWindowWidth()-textWidth)*.5f));
        ImGui::TextDisabled("%s",hint.c_str());
        ImGui::End();
    }
    void card(SDL_Rect q,const std::string& num,const std::string& title,const std::string& sub,SDL_Color accent){fill(ren,q,rgb(0x102b2d));outline(ren,q,accent);fill(ren,{q.x,q.y,7,q.h},accent);text(ren,q.x+28,q.y+20,num,accent,3);text(ren,q.x+90,q.y+14,title,rgb(0xf0f5f2),3);text(ren,q.x+90,q.y+52,sub,rgb(0x89a19f),2);}
    void drawMenu(){}
    void drawGrassBackground(SDL_Rect area){
        fill(ren,area,rgb(0x173d27));
        constexpr int stripe=72;
        for(int x=area.x-stripe;x<area.x+area.w+area.h;x+=stripe){
            SDL_Color shade=((x/stripe)&1)?rgb(0x1b4a2d):rgb(0x1f5331);
            SDL_Vertex v[4]={{{float(x),float(area.y)},shade,{0,0}},{{float(x+stripe),float(area.y)},shade,{0,0}},{{float(x-stripe),float(area.y+area.h)},shade,{0,0}},{{float(x),float(area.y+area.h)},shade,{0,0}}};
            const int indices[6]={0,1,2,1,3,2};SDL_RenderGeometry(ren,nullptr,v,4,indices,6);
        }
        color(ren,rgb(0x285f38,80));for(int y=area.y+28;y<area.y+area.h;y+=96)SDL_RenderDrawLine(ren,area.x,y,area.x+area.w,y);
    }
    void drawTrack(SDL_Rect area,bool nodes,const Transform* supplied=nullptr){
        if(track.points.empty())return;
        Transform t=supplied?*supplied:fitTrack(track,area);
        if(track.points.size()==1){
            if(nodes){auto p=t.screen(track.points.front());fill(ren,{int(p.x)-5,int(p.y)-5,11,11},rgb(0xffc94a));outline(ren,{int(p.x)-7,int(p.y)-7,15,15},rgb(0xffffff));text(ren,int(p.x)+9,int(p.y)-10,"1  START",rgb(0xffc94a),1);}
            return;
        }
        const double coordinateMetres=track.geometryLength/std::max(1.0,track.lengthM);
        auto widthPixels=[&](size_t i,double extra){
            return float(std::max(2.0,(track.widths[i]+extra)*coordinateMetres*t.scale));
        };
        auto grassPixels=[&](size_t i,double extra){
            const double authored=i<track.grassWidths.size()?track.grassWidths[i]:track.widths[i]+10.0;
            return float(std::max(3.0,(std::max(authored,track.widths[i]+6.0)+extra)*coordinateMetres*t.scale));
        };
        for(double extra:{3.5,0.0}){
            SDL_Color surface=extra>0?rgb(0x12341f):rgb(0x2d713d);
            for(size_t i=0;i<track.points.size();++i){size_t j=(i+1)%track.points.size();ribbonSegment(ren,t.screen(track.points[i]),t.screen(track.points[j]),grassPixels(i,extra),grassPixels(j,extra),surface);}
            for(size_t i=0;i<track.points.size();++i)roadJoint(ren,t.screen(track.points[i]),grassPixels(i,extra)*.5f,surface);
        }
        for(double extra:{5.0,0.0}){
            SDL_Color surface=extra>0?rgb(0x20292c):rgb(0x59636a);
            for(size_t i=0;i<track.points.size();++i){
                size_t j=(i+1)%track.points.size();
                ribbonSegment(ren,t.screen(track.points[i]),t.screen(track.points[j]),
                              widthPixels(i,extra),widthPixels(j,extra),surface);
            }
            for(size_t i=0;i<track.points.size();++i)
                roadJoint(ren,t.screen(track.points[i]),widthPixels(i,extra)*.5f,surface);
        }
        auto featureIndex=[&](const char* key)->std::optional<size_t>{if(!track.features.contains(key)||!track.features[key].is_number_integer())return std::nullopt;int value=track.features[key].get<int>();if(value<0||value>=int(track.points.size()))return std::nullopt;return size_t(value);};
        auto featureLine=[&](size_t index,SDL_Color lineColor,float thickness=4){V2 tangent=unit(track.points[(index+1)%track.points.size()]-track.points[(index+track.points.size()-1)%track.points.size()]),n=normal(tangent);double half=track.widths[index]*coordinateMetres*.52;V2 a=track.points[index]-n*half,b=track.points[index]+n*half;ribbonSegment(ren,t.screen(a),t.screen(b),thickness,thickness,lineColor);};
        if(auto start=featureIndex("start_finish"))featureLine(*start,rgb(0xffffff),5);if(track.features.contains("sectors")&&track.features["sectors"].is_array())for(const auto&value:track.features["sectors"])if(value.is_number_integer()){int i=value.get<int>();if(i>=0&&i<int(track.points.size()))featureLine(size_t(i),rgb(0x46e1c1),3);}if(auto i=featureIndex("drs_detection"))featureLine(*i,rgb(0xffc94a),3);if(auto i=featureIndex("drs_entry"))featureLine(*i,rgb(0x4aa3ff),3);if(auto i=featureIndex("drs_exit"))featureLine(*i,rgb(0xb267ff),3);
        if(!track.pitlanePoints.empty()){
            std::vector<V2> lane=track.pitlanePoints;auto entry=featureIndex("pit_entry"),exit=featureIndex("pit_exit");
            if(entry&&exit){auto edgeAnchor=[&](size_t index,V2 toward){V2 tangent=unit(track.points[(index+1)%track.points.size()]-track.points[(index+track.points.size()-1)%track.points.size()]);V2 n=normal(tangent);double side=dot(toward-track.points[index],n)>=0?1.0:-1.0;return track.points[index]+n*(track.widths[index]*coordinateMetres*.5*side);};lane.insert(lane.begin(),edgeAnchor(*entry,lane.front()));lane.push_back(edgeAnchor(*exit,lane.back()));}
            auto pitWidth=[&](size_t laneIndex,bool grass,double extra){if(laneIndex==0||laneIndex+1==lane.size())return float(std::max(3.0,(6.0+extra)*coordinateMetres*t.scale));size_t source=laneIndex-1;double value=grass?(source<track.pitlaneGrassWidths.size()?track.pitlaneGrassWidths[source]:16.0):(source<track.pitlaneWidths.size()?track.pitlaneWidths[source]:6.0);return float(std::max(3.0,(value+extra)*coordinateMetres*t.scale));};
            for(int layer=0;layer<4;++layer){bool grass=layer<2;double extra=(layer%2==0)?(grass?3.0:3.0):0.0;SDL_Color surface=grass?(extra?rgb(0x12341f):rgb(0x2d713d)):(extra?rgb(0x20292c):rgb(0x59636a));for(size_t i=0;i+1<lane.size();++i)ribbonSegment(ren,t.screen(lane[i]),t.screen(lane[i+1]),pitWidth(i,grass,extra),pitWidth(i+1,grass,extra),surface);for(size_t i=0;i<lane.size();++i)roadJoint(ren,t.screen(lane[i]),pitWidth(i,grass,extra)*.5f,surface);}
            if(track.features.contains("pit_start_finish")&&track.features["pit_start_finish"].is_number_integer()){int raw=track.features["pit_start_finish"].get<int>();if(raw>=0&&raw<int(track.pitlanePoints.size())){size_t i=size_t(raw),previous=i?i-1:i,next=std::min(i+1,track.pitlanePoints.size()-1);V2 n=normal(unit(track.pitlanePoints[next]-track.pitlanePoints[previous]));double half=track.pitlaneWidths[i]*coordinateMetres*.55;ribbonSegment(ren,t.screen(track.pitlanePoints[i]-n*half),t.screen(track.pitlanePoints[i]+n*half),4,4,rgb(0xffffff));}}
            if(track.features.contains("pit_boxes")&&track.features["pit_boxes"].is_array()){const int markerSize=nodes?11:clampv(int(std::round(coordinateMetres*t.scale*2.0)),2,7);for(const auto&box:track.features["pit_boxes"])if(box.is_number_integer()){int i=box.get<int>();if(i>=0&&i<int(track.pitlanePoints.size())){auto p=t.screen(track.pitlanePoints[size_t(i)]);fill(ren,{int(p.x)-markerSize/2,int(p.y)-markerSize/2,markerSize,markerSize},rgb(0xffc94a));}}}
        }
        // Draw smoothly curved kerbs following the exact continuous spline contour
        const size_t N = track.points.size();
        if (N >= 3) {
            constexpr double stripeIntervalM = 1.35;
            for (size_t i = 0; i < N; ++i) {
                if (!track.kerbs.count(int(i))) continue;
                size_t j = (i + 1) % N;
                V2 p0 = track.points[(i + N - 1) % N];
                V2 p1 = track.points[i];
                V2 p2 = track.points[j];
                V2 p3 = track.points[(j + 1) % N];
                
                double startDist = track.cumulative[i] * (track.lengthM / std::max(1.0, track.geometryLength));
                double endDist = track.cumulative[i + 1] * (track.lengthM / std::max(1.0, track.geometryLength));
                double segLen = std::max(0.1, endDist - startDist);
                int subdivisions = clampv(int(std::ceil(segLen / stripeIntervalM)), 4, 20);
                
                auto evalKerbEdge = [&](double t, V2& rl, V2& kl, V2& rr, V2& kr) {
                    V2 center = catmull(p0, p1, p2, p3, t);
                    V2 tangent = catmullTangent(p0, p1, p2, p3, t);
                    V2 n = normal(tangent);
                    double wA = track.widths[i], wB = track.widths[j];
                    double halfRoad = (wA * (1.0 - t) + wB * t) * coordinateMetres * 0.5;
                    double kerbExtra = 2.0 * coordinateMetres;
                    rl = center + n * halfRoad;
                    kl = center + n * (halfRoad + kerbExtra);
                    rr = center - n * halfRoad;
                    kr = center - n * (halfRoad + kerbExtra);
                };
                
                V2 prev_rl, prev_kl, prev_rr, prev_kr;
                evalKerbEdge(0.0, prev_rl, prev_kl, prev_rr, prev_kr);
                
                for (int sub = 0; sub < subdivisions; ++sub) {
                    double t0 = double(sub) / subdivisions;
                    double t1 = double(sub + 1) / subdivisions;
                    double subDist = startDist + (endDist - startDist) * (t0 + 0.0001);
                    int band = int(std::floor(subDist / stripeIntervalM));
                    SDL_Color kerbCol = (band % 2 == 0) ? rgb(0xd9202a) : rgb(0xf6f6f2);
                    
                    V2 curr_rl, curr_kl, curr_rr, curr_kr;
                    evalKerbEdge(t1, curr_rl, curr_kl, curr_rr, curr_kr);
                    
                    // Left side curved kerb quad
                    SDL_Vertex vL[4] = {
                        {t.screen(prev_rl), kerbCol, {0,0}},
                        {t.screen(curr_rl), kerbCol, {0,0}},
                        {t.screen(curr_kl), kerbCol, {0,0}},
                        {t.screen(prev_kl), kerbCol, {0,0}}
                    };
                    const int indices[6] = {0, 1, 2, 0, 2, 3};
                    SDL_RenderGeometry(ren, nullptr, vL, 4, indices, 6);
                    
                    // Right side curved kerb quad
                    SDL_Vertex vR[4] = {
                        {t.screen(prev_rr), kerbCol, {0,0}},
                        {t.screen(curr_rr), kerbCol, {0,0}},
                        {t.screen(curr_kr), kerbCol, {0,0}},
                        {t.screen(prev_kr), kerbCol, {0,0}}
                    };
                    SDL_RenderGeometry(ren, nullptr, vR, 4, indices, 6);
                    
                    // Outer edge joint circle for round smooth finish
                    roadJoint(ren, t.screen(curr_kl), float(0.8 * coordinateMetres * t.scale), kerbCol);
                    roadJoint(ren, t.screen(curr_kr), float(0.8 * coordinateMetres * t.scale), kerbCol);
                    
                    prev_rl = curr_rl;
                    prev_kl = curr_kl;
                    prev_rr = curr_rr;
                    prev_kr = curr_kr;
                }
            }
        }
        if(nodes){
            const size_t labelStep=track.points.size()>140?std::max<size_t>(1,track.points.size()/70):1;
            for(size_t i=0;i<track.points.size();++i){auto p=t.screen(track.points[i]);SDL_Color marker=rgb(0x46e1c1);const char* tag=nullptr;if(featureIndex("start_finish")==i){marker=rgb(0xffc94a);tag="START";}if(featureIndex("pit_entry")==i){marker=rgb(0xf97316);tag="PIT IN";}if(featureIndex("pit_exit")==i){marker=rgb(0x22c55e);tag="PIT OUT";}fill(ren,{int(p.x)-3,int(p.y)-3,7,7},marker);if(i%labelStep==0)text(ren,int(p.x)+5,int(p.y)-8,std::to_string(i+1),marker,1);if(tag)text(ren,int(p.x)+5,int(p.y)+3,tag,marker,1);}
            for(size_t i=0;i<track.pitlanePoints.size();++i){auto p=t.screen(track.pitlanePoints[i]);fill(ren,{int(p.x)-4,int(p.y)-4,9,9},rgb(0x4aa3ff));text(ren,int(p.x)+6,int(p.y)-6,"P"+std::to_string(i+1),rgb(0x9cc9ff),1);}
        }
    }
    void drawTrackEditor(){text(ren,35,82,"TRACK STUDIO",rgb(0x46e1c1),3);SDL_Rect clip{30,115,1180,745};SDL_RenderSetClipRect(ren,&clip);drawGrassBackground(clip);Transform t=editorTransform();drawTrack(clip,true,&t);SDL_RenderSetClipRect(ren,nullptr);text(ren,48,830,"WHEEL ZOOM  /  MIDDLE-DRAG PAN  /  LEFT-DRAG NODE  /  LEFT+WHEEL WIDTH",rgb(0x76928f),2);}
    std::vector<std::pair<size_t,std::string>> editorLines()const{std::vector<std::pair<size_t,std::string>> out;size_t start=0;while(start<=editorSource.size()){size_t end=editorSource.find('\n',start);out.push_back({start,editorSource.substr(start,end==std::string::npos?std::string::npos:end-start)});if(end==std::string::npos)break;start=end+1;}return out;}
    void drawAlgorithm(){fill(ren,{40,100,1050,750},rgb(0x081217));outline(ren,{40,100,1050,750},rgb(0x285159));text(ren,55,76,hybrid?"ALGORITHM LAB  /  HYBRID":"ALGORITHM LAB  /  ICE",hybrid?rgb(0xb267ff):rgb(0xff7f3f),2);auto lines=editorLines();auto[a,b]=selection();for(int row=0;row<37;++row){int li=row+editorScroll;if(li>=int(lines.size()))break;size_t begin=lines[li].first;std::string line=lines[li].second;int y=120+row*18;text(ren,55,y,std::to_string(li+1),rgb(0x4e6a70),2);if(a!=b){size_t sa=std::max(a,begin),sb=std::min(b,begin+line.size());if(sb>sa)fill(ren,{85+int(sa-begin)*12,y-2,int(sb-sa)*12,17},rgb(0x245c78));}text(ren,86,y,line,rgb(0xc8dbd5),2,82);if(cursor>=begin&&cursor<=begin+line.size()&&((SDL_GetTicks()/500)%2)==0)fill(ren,{85+int(cursor-begin)*12,y-2,2,17},rgb(0x46e1c1));}fill(ren,{1120,100,440,750},rgb(0x102628));text(ren,1150,135,"EDITOR SHORTCUTS",rgb(0x4aa3ff),2);text(ren,1150,180,"CTRL/CMD + C  COPY\nCTRL/CMD + X  CUT\nCTRL/CMD + V  PASTE\nCTRL/CMD + A  SELECT ALL\nCTRL/CMD + Z  UNDO\nCTRL/CMD + Y  REDO\nCTRL/CMD + S  SAVE\nTAB           INDENT",rgb(0xaac0bc),2);text(ren,1150,400,"SAVES TO NATIVE DATA.\nTHE ORIGINAL PYTHON\nFILES ARE NEVER CHANGED.",rgb(0xffc94a),2);}
    void drawReplaySetup(){text(ren,70,95,"REPLAY THEATRE",rgb(0xb267ff),3);fill(ren,{70,150,1000,500},rgb(0x102628));for(size_t i=0;i<replayFiles.size()&&i<10;++i){SDL_Color c=i==replayIndex?rgb(0x46e1c1):rgb(0x8ca4a0);if(i==replayIndex)fill(ren,{90,180+int(i)*42,920,34},rgb(0x19413f));text(ren,110,190+int(i)*42,replayFiles[i].stem().string(),c,2,65);}fill(ren,{70,690,360,65},rgb(0x254f4c));text(ren,105,713,"PLAY SELECTED REPLAY",rgb(0xffffff),2);text(ren,1120,180,"J  REWIND\nK  PAUSE\nL  FAST FORWARD\nLEFT/RIGHT  SEEK\nUP/DOWN  CAMERA",rgb(0xaac0bc),2);}
    void drawCarScreen(SDL_FPoint p,double angle,SDL_Color c,bool hybridEra,bool selected,double cameraScale){
        SDL_Texture* texture=hybridEra?hybridCarMaster:iceCarMaster;
        const auto spriteSize=carSpriteDimensions(cameraScale);
        const float carLength=spriteSize.first,carWidth=spriteSize.second;
        SDL_FRect destination{p.x-carLength*.5f,p.y-carWidth*.5f,carLength,carWidth};
        if(texture){
            SDL_FRect shadow=destination;shadow.x+=2.5f;shadow.y+=3.5f;SDL_SetTextureColorMod(texture,12,16,16);SDL_SetTextureAlphaMod(texture,125);SDL_RenderCopyExF(ren,texture,nullptr,&shadow,angle*180.0/PI,nullptr,SDL_FLIP_NONE);
            const Uint8 lr=Uint8(std::min(255,int(c.r*.78f+55))),lg=Uint8(std::min(255,int(c.g*.78f+55))),lb=Uint8(std::min(255,int(c.b*.78f+55)));
            SDL_SetTextureColorMod(texture,lr,lg,lb);SDL_SetTextureAlphaMod(texture,255);SDL_RenderCopyExF(ren,texture,nullptr,&destination,angle*180.0/PI,nullptr,SDL_FLIP_NONE);SDL_SetTextureColorMod(texture,255,255,255);
        } else {
            const double lengthScale=carLength/CAR_LENGTH_M,widthScale=carWidth/CAR_WIDTH_M;double cs=std::cos(angle),sn=std::sin(angle);auto pt=[&](double x,double y){double sx=x*lengthScale,sy=y*widthScale;return SDL_FPoint{float(p.x+sx*cs-sy*sn),float(p.y+sx*sn+sy*cs)};};std::array<SDL_FPoint,5> q={pt(2.8,0),pt(-2.8,-1),pt(-2,-.45),pt(-2,.45),pt(-2.8,1)};color(ren,c);SDL_RenderDrawLinesF(ren,q.data(),int(q.size()));
        }
        if(selected){color(ren,rgb(0x46e1c1));constexpr int segments=24;std::array<SDL_FPoint,segments+1> ring{};float radius=float(std::max(12.0,std::round(2.2*cameraScale)));for(int i=0;i<=segments;++i){double a=i*PI*2/segments;ring[size_t(i)]={p.x+float(std::cos(a)*radius),p.y+float(std::sin(a)*radius)};}SDL_RenderDrawLinesF(ren,ring.data(),int(ring.size()));}
    }
    Transform simulationTransform(SDL_Rect world)const{Transform t=fitTrack(track,world);V2 target{};bool found=false;if(mode!=Mode::Replay&&!cars.empty()&&focus<int(cars.size())){target=cars[size_t(focus)].position;found=true;}else if(mode==Mode::Replay&&!replay.empty()){auto it=std::lower_bound(replay.begin(),replay.end(),replayTime,[](const ReplayFrame&f,double v){return f.time<v;});const auto&frame=it==replay.end()?replay.back():*it;if(!frame.cars.empty()){int i=clampv(focus,0,int(frame.cars.size())-1);target={frame.cars[size_t(i)].x,frame.cars[size_t(i)].y};found=true;}}if(found){t.scale=cameraZoom;t.ox=world.x+world.w*.5-target.x*t.scale;t.oy=world.y+world.h*.5-target.y*t.scale;}return t;}
    void drawSimulation(){
        SDL_Rect world{465,90,1110,770};
        SDL_RenderSetClipRect(ren,&world);
        drawGrassBackground(world);
        Transform t=simulationTransform(world);
        drawTrack(world,false,&t);
        if(mode==Mode::Race)drawGridBoxes(t,int(cars.size()));
        if(mode==Mode::Replay)drawReplayCars(t);
        else for(size_t i=0;i<cars.size();++i)if(!cars[i].removed)drawCarScreen(t.screen(cars[i].position),cars[i].angle,cars[i].col,cars[i].hybrid,int(i)==focus,t.scale);
        SDL_RenderSetClipRect(ren,nullptr);
        // Backdrop only needs to cover the timing tower's footprint now that the minimap lives in the top-right corner of the track view (see drawMiniMap); it used to run all the way down and collide with the minimap and the telemetry panel below it.
        fill(ren,{18,92,542,614},rgb(0x102628));
        if(mode==Mode::Replay){text(ren,35,108,"REPLAY",rgb(0xf2f6f4),2);}
        drawMiniMap();
        if(mode==Mode::Race&&countdown>0)drawLights();
        if(paused)text(ren,870,420,"PAUSED",rgb(0xffc94a),4);
    }
    void drawTower(){if(mode==Mode::Replay)return;std::vector<int> order(cars.size());for(size_t i=0;i<cars.size();++i)order[i]=int(i);std::sort(order.begin(),order.end(),[&](int a,int b){return cars[a].lap==cars[b].lap?cars[a].s>cars[b].s:cars[a].lap>cars[b].lap;});for(size_t row=0;row<order.size()&&row<20;++row){int i=order[row],y=172+int(row)*31;if(i==focus)fill(ren,{26,y-5,306,27},rgb(0x214449));fill(ren,{35,y,8,14},cars[i].col);text(ren,52,y,std::to_string(row+1),rgb(0xdce8e4),2);text(ren,78,y,cars[i].name,rgb(0xdce8e4),2,8);std::ostringstream s;s<<int(cars[i].speed)<<" G"<<cars[i].gear;text(ren,190,y,s.str(),rgb(0x90aaa5),2);if(cars[i].hybrid)text(ren,268,y,std::to_string(int(cars[i].battery)),cars[i].regen?rgb(0x4aa3ff):cars[i].deploying?rgb(0xffc94a):rgb(0x90aaa5),2);}}
    // Paints one starting box per grid slot using the exact same row/column formula startRace() uses to place the cars, so the boxes always match however many cars (and whichever grid geometry) the race was actually configured with.
    void drawGridBoxes(const Transform&t,int carCount){
        if(carCount<=0||track.points.empty())return;
        const double coordinateMetres=track.geometryLength/std::max(1.0,track.lengthM);
        for(int i=0;i<carCount;++i){
            const int row=i/2,column=i%2;
            const double targetDistance=row*(CAR_LENGTH_M+8.0)+column*4.0;
            const double s=std::fmod(track.lengthM-targetDistance+track.lengthM,track.lengthM);
            const double halfWidth=clampv(track.widthAt(0)*.20,CAR_WIDTH_M*.5+.55,std::max(CAR_WIDTH_M*.5+.55,track.widthAt(0)*.5-CAR_WIDTH_M*.5-.75));
            const double lateral=column?-halfWidth:halfWidth;
            auto[base,tangent]=track.at(s);
            V2 forward=tangent,side=normal(tangent);
            V2 center=base+side*(lateral*coordinateMetres);
            const double boxLen=(CAR_LENGTH_M+1.6)*coordinateMetres*.5,boxWid=(CAR_WIDTH_M+1.2)*coordinateMetres*.5;
            V2 frontLeft=center+forward*boxLen+side*boxWid,backLeft=center-forward*boxLen+side*boxWid;
            V2 backRight=center-forward*boxLen-side*boxWid,frontRight=center+forward*boxLen-side*boxWid;
            SDL_FPoint pts[4]={t.screen(backLeft),t.screen(frontLeft),t.screen(frontRight),t.screen(backRight)};
            SDL_SetRenderDrawColor(ren,255,255,255,185);
            SDL_RenderDrawLinesF(ren,pts,4);
            SDL_FPoint labelAt=t.screen(center-forward*(boxLen*.35));
            text(ren,int(labelAt.x)-4,int(labelAt.y)-6,std::to_string(i+1),rgb(0xffffff),1);
        }
    }
    void drawMiniMap(){
        // Top-right corner of the track view: clear of the timing tower, session-control bar and telemetry panel at every window size, unlike the old bottom-left slot which crowded/overlapped those ImGui windows.
        SDL_Rect q{1385,150,195,175};fill(ren,q,rgb(0x081619));SDL_RenderSetClipRect(ren,&q);Transform t=fitTrack(track,q,15);drawTrack(q,false,&t);if(mode!=Mode::Replay){for(size_t i=0;i<cars.size();++i){if(cars[i].removed)continue;auto p=t.screen(cars[i].position);int radius=int(i)==focus?4:2;fill(ren,{int(p.x)-radius,int(p.y)-radius,radius*2+1,radius*2+1},cars[i].col);if(int(i)==focus)outline(ren,{int(p.x)-radius-2,int(p.y)-radius-2,radius*2+5,radius*2+5},rgb(0xffffff));}}else if(!replay.empty()){auto it=std::lower_bound(replay.begin(),replay.end(),replayTime,[](const ReplayFrame&f,double v){return f.time<v;});const auto&f=it==replay.end()?replay.back():*it;for(size_t i=0;i<f.cars.size();++i){auto p=t.screen({f.cars[i].x,f.cars[i].y});int radius=int(i)==focus?4:2;fill(ren,{int(p.x)-radius,int(p.y)-radius,radius*2+1,radius*2+1},palette[i%palette.size()]);}}SDL_RenderSetClipRect(ren,nullptr);}
    void drawLights(){
        int lit=clampv(5-int(std::ceil(countdown)),0,5);
        bool active=countdown>0;
        fill(ren,{670,110,480,135},rgb(0x0a1215));
        outline(ren,{670,110,480,135},rgb(0x3e4844));
        std::string title=active?"RACE START  •  "+std::to_string(std::max(1,int(std::ceil(countdown)))):"LIGHTS OUT";
        text(ren,850,122,title,active?rgb(0xffffff):rgb(0x46e1c1),2);
        for(int i=0;i<5;++i){
            SDL_Rect pod{695+i*90,148,74,80};
            fill(ren,pod,rgb(0x0a0d0d));
            outline(ren,pod,rgb(0x303734));
            SDL_Color lightCol=(i<lit)?rgb(0xff232d):(active?rgb(0x481217):rgb(0x192d27));
            roadJoint(ren,{float(pod.x+pod.w/2),float(pod.y+pod.h/2)},24,lightCol);
            if(i<lit) roadJoint(ren,{float(pod.x+pod.w/2),float(pod.y+pod.h/2)},12,rgb(0xff7e74));
        }
    }
    void drawReplayCars(const Transform&t){if(replay.empty())return;auto it=std::lower_bound(replay.begin(),replay.end(),replayTime,[](const ReplayFrame&f,double v){return f.time<v;});size_t idx=it==replay.end()?replay.size()-1:size_t(it-replay.begin());auto&f=replay[idx];for(size_t i=0;i<f.cars.size();++i)if(!f.cars[i].removed)drawCarScreen(t.screen({f.cars[i].x,f.cars[i].y}),f.cars[i].angle,f.cars[i].col,f.cars[i].generation=="Hybrid",int(i)==focus,t.scale);for(size_t i=0;i<f.cars.size()&&i<20;++i){int y=172+int(i)*31;if(int(i)==focus)fill(ren,{26,y-5,306,27},rgb(0x214449));text(ren,42,y,std::to_string(i+1),rgb(0xe5efeb),2);text(ren,74,y,f.cars[i].name,rgb(0xe5efeb),2,9);text(ren,200,y,std::to_string(int(f.cars[i].speed))+" G"+std::to_string(f.cars[i].gear),rgb(0x8fa7a2),2);}std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<replayTime<<" / "<<replay.back().time<<" S  X"<<replaySpeed;text(ren,800,830,s.str(),rgb(0xffc94a),2);}
};

bool smoke() {
    const auto distantSprite=carSpriteDimensions(MIN_CAMERA_ZOOM),normalSprite=carSpriteDimensions(DEFAULT_CAMERA_ZOOM);
    if(distantSprite.first!=18||distantSprite.second!=8||normalSprite.first!=45||normalSprite.second!=16){std::cerr<<"Python-compatible car sprite scaling failed\n";return false;}
    auto tracks=filesFor("tracks",".json"); if(tracks.empty()){std::cerr<<"No compatible tracks found\n";return false;}
    Track t;if(!t.load(tracks.front())||t.lengthM<=0){std::cerr<<"Track load failed\n";return false;}
    bool checkedPitlane=false;for(const auto&path:tracks){Track candidate;if(candidate.load(path)&&!candidate.pitlanePoints.empty()){json encoded=candidate.toJson();if(encoded["pitlane_points"].size()!=candidate.pitlanePoints.size()||!candidate.features.contains("pit_entry")||!candidate.features.contains("pit_exit")){std::cerr<<"Pitlane schema failed\n";return false;}checkedPitlane=true;break;}}
    auto brains=filesFor("brains",".json"); if(!brains.empty()){Brain b=Brain::load(brains.front());std::mt19937 r(1);auto m=b.mutate(r);(void)m;}
    for(const auto& path:filesFor("algorithms",".fai")){fai::Program program;if(!program.compile(readFile(path))){std::cerr<<"Algorithm compile failed: "<<path.filename()<<": "<<program.error()<<'\n';return false;}auto output=program.run({},{});if(!std::isfinite(output.steering)||!std::isfinite(output.throttle)){std::cerr<<"Algorithm produced invalid controls: "<<path.filename()<<'\n';return false;}}
    auto [p,d]=t.at(t.lengthM*.5);if(!std::isfinite(p.x)||std::abs(length(d)-1)>.01)return false;
    std::cout<<"Loaded "<<tracks.size()<<" tracks; "<<t.name<<" = "<<t.lengthM<<" m; pitlane "<<(checkedPitlane?"verified":"not present")<<"\n";return true;
}
bool simulationSmoke(){
    auto tracks=filesFor("tracks",".json");auto algorithms=filesFor("algorithms",".fai");if(tracks.empty()||algorithms.empty())return false;Track track;bool loaded=false;for(const auto&path:tracks)if(path.filename()=="oval_002.json"||path.filename()=="thespa4.json"){loaded=track.load(path);if(loaded)break;}if(!loaded)loaded=track.load(tracks.front());fs::path controller;for(const auto&path:algorithms)if(path.filename()=="ice_controller.fai")controller=path;if(controller.empty())controller=algorithms.front();Brain brain;brain.name="SIMULATION TEST";brain.setSource(readFile(controller));if(!brain.program.valid()){std::cerr<<brain.program.error()<<'\n';return false;}std::mt19937 testRng(7);std::vector<Car>field(6);for(size_t i=0;i<field.size();++i){Car&car=field[i];car.brain=i?brain.mutate(testRng,.02):brain;car.s=std::fmod(track.lengthM-double(i)*14+track.lengthM,track.lengthM);auto[p,t]=track.at(car.s);car.position=p+normal(t)*((i%2)?1.25:-1.25);car.angle=angleOf(t);car.previousProgress=car.s;car.raceDistance=-double(i)*14;car.name="TEST "+std::to_string(i+1);}
    auto started=std::chrono::steady_clock::now();for(int frame=0;frame<600;++frame)for(Car&car:field)car.update(track,1.0/60.0,frame/60.0,false,0,0,false);double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();for(const Car&car:field)if(!std::isfinite(car.position.x)||!std::isfinite(car.position.y)||!std::isfinite(car.speed)||!std::isfinite(car.fitness))return false;if(field.front().forwardDistance<5){std::cerr<<"Controller did not advance: "<<field.front().forwardDistance<<" m\n";return false;}
    fs::path hybridController;for(const auto&path:algorithms)if(path.filename()=="hybrid_controller.fai")hybridController=path;if(!hybridController.empty()){Brain actualHybrid;actualHybrid.setSource(readFile(hybridController));Car policyCar;policyCar.brain=actualHybrid;policyCar.hybrid=true;policyCar.position=track.at(0).first;policyCar.angle=angleOf(track.at(0).second);policyCar.previousProgress=0;for(int frame=0;frame<600;++frame)policyCar.update(track,1.0/60.0,frame/60.0,false,0,0,false);if(policyCar.forwardDistance<5||!std::isfinite(policyCar.battery)){std::cerr<<"Bundled Hybrid controller failed native simulation\n";return false;}}
    Brain energyBrain;energyBrain.setSource("steering = 0.0\nthrottle = 1.0\nbrake = 0.0\novertake = 1.0\nrecharge = 0.0\n");Car hybridCar;hybridCar.brain=energyBrain;hybridCar.hybrid=true;hybridCar.battery=80;auto[startPoint,startTangent]=track.at(0);hybridCar.position=startPoint;hybridCar.angle=angleOf(startTangent);hybridCar.previousProgress=0;for(int frame=0;frame<60;++frame)hybridCar.update(track,1.0/60.0,frame/60.0,false,0,0,false);double deployedBattery=hybridCar.battery;if(!(deployedBattery<80)){std::cerr<<"Hybrid deployment did not drain battery\n";return false;}energyBrain.setSource("steering = 0.0\nthrottle = 1.0\nbrake = 0.0\novertake = 0.0\nrecharge = 1.0\n");hybridCar.brain=energyBrain;hybridCar.battery=10;for(int frame=0;frame<60;++frame)hybridCar.update(track,1.0/60.0,1+frame/60.0,false,0,0,false);if(!(hybridCar.battery>10)){std::cerr<<"Hybrid recharge did not restore battery\n";return false;}
    Car redFlagCar;redFlagCar.brain=brain;redFlagCar.velocity={5.0,0.0};redFlagCar.health=45.0;redFlagCar.wear=0.85;redFlagCar.update(track,1.0/60.0,0,false,0,0,false,nullptr,true);if(redFlagCar.redFlagPitStopped){std::cerr<<"Red flag pit stopped prematurely\n";return false;}redFlagCar.redFlagPitStopped=true;redFlagCar.update(track,1.0/60.0,0,false,0,0,false,nullptr,true);if(length(redFlagCar.velocity)>1e-6||redFlagCar.speed>1e-6){std::cerr<<"Red flag stopped car moved\n";return false;}
    std::cout<<"Simulation: "<<field.size()<<" cars x 600 frames in "<<elapsed<<" s; leader "<<field.front().forwardDistance<<" m, "<<field.front().speed<<" km/h; hybrid & red flag verified\n";return true;
}
} // namespace

int main(int argc,char**argv){std::string arg=argc>1?argv[1]:"";if(arg=="--smoke-test")return smoke()?0:1;if(arg=="--simulation-test")return simulationSmoke()?0:1;App app;if(!app.init())return 1;if((arg.find("small")!=std::string::npos||arg.find("screenshot")!=std::string::npos)&&arg.find("live")==std::string::npos)app.resizeForTest(1100,680);else if(arg.find("live")!=std::string::npos)app.resizeForTest(1600,900);if(arg=="--ui-screenshot-editor"||arg=="--ui-track-studio")app.openWorkspace(0);if(arg=="--ui-training-setup")app.openWorkspace(1);if(arg=="--ui-screenshot-race"||arg=="--ui-race-setup")app.openWorkspace(2);if(arg=="--ui-hotlap-setup")app.openWorkspace(3);
    int frameLimit=arg.rfind("--ui-",0)==0?3:-1;
    if(arg=="--ui-live-race"){app.startRace();frameLimit=180;}
    if(arg=="--ui-live-training"){app.startTraining();frameLimit=180;}
    app.run(frameLimit);if((arg.find("screenshot")!=std::string::npos||arg.find("live")!=std::string::npos)&&!app.saveScreenshot("/tmp/formula_ai_cpp_ui.bmp"))return 2;return 0;}
