#include "StereoMatch.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <random>
#include <omp.h>

// Same explicit debug-friend convention as the existing pybind interface.
struct DebugAccess_StereoMatch {
#ifndef LEGACY_BASELINE
  static bool line(const StereoMatch& m,int c,const Line3D& r,Line2D& l) { return m.makeMatchingLine(c,r,l); }
  static bool raw(const StereoMatch& m,int c,const Line3D& r,Line2D& l) { return m.makeLine2DFromLOS3D(c,r,l); }
  static bool mode(const StereoMatch& m) { return m._pinhole_matching; }
  static const auto& valid(const StereoMatch& m) { return m._matching_valid; }
  static const auto& pixels(const StereoMatch& m) { return m._matching_pixels; }
  static void enumerate(const StereoMatch& m,const std::vector<Line3D>& rays,int c,
      const std::vector<int>& chosen,const std::vector<Pt2D>& q,std::vector<int>& ids) {
    m.enumerateCandidatesOnCam(rays,c,chosen,q,ids);
  }
  static bool reciprocal(const StereoMatch& m,int c,const Pt2D& q,
      const std::vector<int>& chosen,const std::vector<Pt2D>& p) {
    return m.checkMatchingBackProjection(c,q,chosen,p);
  }
#else
  static bool mode(const StereoMatch&) { return false; }
  static bool line(const StereoMatch& m,int c,const Line3D& r,Line2D& l) { return m.makeLine2DFromLOS3D(c,r,l); }
  static bool raw(const StereoMatch& m,int c,const Line3D& r,Line2D& l) { return m.makeLine2DFromLOS3D(c,r,l); }
#endif
};
// Parser itself is also checked for the legacy empty coefficient field.
void loadDistCoeff(const std::string&, PinholeParam&);

namespace {
int failures = 0, checks = 0;
void expect(bool yes, const std::string &what) {
  ++checks;
  if (!yes) { ++failures; std::cout << "FAIL " << what << '\n'; }
}
using Coeff = std::vector<double>;
using Obs = std::vector<std::vector<std::unique_ptr<Object2D>>>;
struct Pose { double angle; Pt3D center; double fx, fy, cx, cy; };
const std::array<Pose,4> poses{{
  {0.12, {-2,-0.4,0}, 820,910,483,397},
  {-0.11,{2,0.7,0.3}, 870,790,512,381},
  {0.04,{-0.2,2,-0.3}, 760,880,477,413},
  {-0.06,{1,-2,0.6}, 930,810,503,389}}};
// Independent oracle: explicit y rotation and Brown-Conrady/rational/prism
// forward map. Never calls production projection to generate observations.
Pt3D frame(const Pose &p, const Pt3D &X) {
  const double x=X[0]-p.center[0], y=X[1]-p.center[1], z=X[2]-p.center[2];
  return {std::cos(p.angle)*x+std::sin(p.angle)*z, y,
          -std::sin(p.angle)*x+std::cos(p.angle)*z};
}
Pt2D oracle(const Pose &p, const Pt3D &X, const Coeff &d) {
  auto v=frame(p,X); double x=v[0]/v[2], y=v[1]/v[2];
  auto k=[&](int i){ return i<(int)d.size()?d[i]:0.; };
  double r=x*x+y*y, r4=r*r, r6=r4*r;
  double radial=(1+k(0)*r+k(1)*r4+k(4)*r6)/(1+k(5)*r+k(6)*r4+k(7)*r6);
  return {p.fx*(x*radial+2*k(2)*x*y+k(3)*(r+2*x*x)+k(8)*r+k(9)*r4)+p.cx,
          p.fy*(y*radial+k(2)*(r+2*y*y)+2*k(3)*x*y+k(10)*r+k(11)*r4)+p.cy};
}
std::shared_ptr<PinholeCamera> camera(const Pose &p, const Coeff &d) {
  auto c=std::make_shared<PinholeCamera>();
  c->setImageSize(800,1000); c->setIntrinsics(p.fx,p.fy,p.cx,p.cy,d);
  double a=p.angle;
  Pt3D t{-std::cos(a)*p.center[0]-std::sin(a)*p.center[2],-p.center[1],
          std::sin(a)*p.center[0]-std::cos(a)*p.center[2]};
  c->setExtrinsics({0,a,0},t); c->commitUpdate(); return c;
}
std::vector<Pt3D> points() {
  // Fixed, noncoplanar, well-separated off-axis scene (scene scale ~15).
  return {{-1.7,-1.8,9.2},{1.7,1.4,10.8},{-0.8,2.0,13.1},
          {2.2,-1.1,11.7},{0.3,-0.4,8.7},{-1.5,0.7,14.2}};
}
TracerConfig config(int build) {
  TracerConfig cfg; cfg._sm_param.match_cam_count=build;
  cfg._sm_param.tol_2d_px=0.8; cfg._sm_param.tol_3d_mm=1e-4;
  cfg._sm_param.limit={-100,100,-100,100,-100,100}; return cfg;
}
void scene(const Coeff &d, int build, const std::string &name) {
  std::vector<std::shared_ptr<Camera>> cams;
  Obs obs(4); auto X=points(); double shift=0, proj_err=0, ray_err=0;
  for(int c=0;c<4;++c) {
    cams.push_back(camera(poses[c],d));
    for(auto &x:X) {
      auto q=oracle(poses[c],x,d); obs[c].push_back(std::make_unique<Tracer2D>(q));
      auto actual=cams[c]->project(x);
      expect(bool(actual),name+" project status");
      if(actual) proj_err=std::max(proj_err,(actual.value()-q).norm());
      auto ray=cams[c]->lineOfSight(q); expect(bool(ray),name+" ray status");
      if(ray) {
        auto truth=x-poses[c].center; truth=truth/truth.norm();
        ray_err=std::max(ray_err,myMATH::cross(truth,ray.value().unit_vector).norm());
      }
      shift=std::max(shift,(q-oracle(poses[c],x,Coeff(5,0))).norm());
    }
    obs[c].push_back(std::make_unique<Tracer2D>(Pt2D{71.+19*c,109.+37*c}));
  }
  expect(proj_err<=1e-6,name+" independent projection <=1e-6 px");
  expect(ray_err<=1e-7,name+" independent ray <=1e-7 rad");
  // Independent forward-Jacobian check throughout the convex image region
  // containing this fixture. Positive symmetric part gives a monotone inverse
  // region, well away from the separate pathological inverse tests.
  double bound=0;
  for(auto& p:poses) for(auto& x:X) { auto f=frame(p,x);
    bound=std::max({bound,std::abs(f[0]/f[2]),std::abs(f[1]/f[2])}); }
  Pose normalized{0,{0,0,0},1,1,0,0}; double min_eigen=1e100;
  for(int i=-10;i<=10;++i) for(int j=-10;j<=10;++j) {
    double x=bound*i/10,y=bound*j/10,h=1e-6;
    auto a=(oracle(normalized,{x+h,y,1},d)-oracle(normalized,{x-h,y,1},d))/(2*h);
    auto b=(oracle(normalized,{x,y+h,1},d)-oracle(normalized,{x,y-h,1},d))/(2*h);
    double off=(a[1]+b[0])/2;
    min_eigen=std::min(min_eigen,(a[0]+b[1]-std::hypot(a[0]-b[1],2*off))/2);
  }
  expect(min_eigen>0.1,name+" independently checked invertible region");
  bool nonzero=false; for(double k:d) nonzero |= k!=0;
  if(nonzero) expect(shift>1e-7,name+" nonzero oracle displacement");
  auto cfg=config(build); StereoMatch matcher(cams,obs,cfg);
  auto result=matcher.match();
  expect(result.size()==X.size(),name+" recall/count "+std::to_string(result.size())+"/6");
  std::set<int> found; double err=0;
  for(auto &r:result) {
    int id=-1; double best=1e100;
    for(int i=0;i<(int)X.size();++i) { double e=(r->_pt_center-X[i]).norm(); if(e<best){best=e;id=i;} }
    err=std::max(err,best); expect(best<=1e-5*15,name+" 3D fidelity");
    expect(found.insert(id).second,name+" no duplicate/ghost");
    for(int c=0;c<4;++c) if(r->_obj2d_list[c])
      expect((r->_obj2d_list[c]->_pt_center-obs[c][id]->_pt_center).norm()==0,name+" original pixels/identity");
  }
  auto repeat=matcher.match(); expect(repeat.size()==result.size(),name+" repeat count");
  std::cout<<"SCENE "<<name<<" build="<<build<<" matches="<<result.size()<<" max3d="<<err
           <<" maxproj="<<proj_err<<" maxray="<<ray_err<<" shift="<<shift<<'\n';
}
void loader() {
  for(int n:{4,5,8,12}) {
    Coeff d(n,0); d[0]=-0.08;
    auto c=camera(poses[0],d); std::string file="negative_"+std::to_string(n)+".txt";
    expect(bool(c->saveParameters(file)),"save negative fixture");
    auto read=CameraFactory::loadFromFile(file); expect(bool(read),"load negative fixture");
    if(!read) continue;
    auto loaded=std::dynamic_pointer_cast<PinholeCamera>(read.value());
    expect(loaded && loaded->param().is_distorted,"negative-only loader activation n="+std::to_string(n));
    auto q=loaded->project(points()[0]);
    expect(q && (q.value()-oracle(poses[0],points()[0],d)).norm()<1e-6,"negative loader projection");
    if(q) {
      auto ray=loaded->lineOfSight(q.value()); auto truth=points()[0]-poses[0].center;
      truth=truth/truth.norm();
      expect(ray && myMATH::cross(truth,ray.value().unit_vector).norm()<1e-7,"negative loader ray");
    }
  }
  for(double k:{0.,SMALLNUMBER,-SMALLNUMBER,2*SMALLNUMBER,-2*SMALLNUMBER}) {
    Coeff d(5,0); d[0]=k;
    auto c=camera(poses[0],d); c->saveParameters("threshold.txt");
    auto r=CameraFactory::loadFromFile("threshold.txt");
    expect(r && std::dynamic_pointer_cast<PinholeCamera>(r.value())->param().is_distorted==c->param().is_distorted,
           "loader/setter activation threshold "+std::to_string(k));
  }
  PinholeParam empty; loadDistCoeff("",empty);
  expect(!empty.is_distorted && empty.n_dist_coeff==0 && empty.dist_coeff.empty(),"legacy empty coefficients");
  bool caught=false; try { loadDistCoeff("-0.08,0,0",empty); } catch(const std::exception&) { caught=true; }
  expect(caught,"invalid nonzero coefficient length rejects");
  loadDistCoeff("0,0,0",empty);
  expect(!empty.is_distorted && empty.n_dist_coeff==0,"legacy all-zero short coefficient field");
  auto ref=std::make_shared<RefractionPinholeCamera>();
  ref->setImageSize(800,1000); ref->setIntrinsics(820,910,483,397,{-0.08,0,0,0,0});
  ref->setExtrinsics({0,0,0},{0,0,0});
  ref->setRefraction({0,0,3},{0,0,1},{1,1.5,1},{0.5});
  ref->setSolverOptions(1e-9,100,0.1); ref->commitUpdate();
  ref->saveParameters("negative_refractive.txt");
  auto loaded=CameraFactory::loadFromFile("negative_refractive.txt");
  expect(loaded && std::dynamic_pointer_cast<RefractionPinholeCamera>(loaded.value())->param().is_distorted,
         "shared refractive negative loader activation");
}

double distance(const Pt2D& q,const Line2D& l) {
  const double dx=q[0]-l.pt[0],dy=q[1]-l.pt[1];
  return std::abs(dx*l.unit_vector[1]-dy*l.unit_vector[0])/
         std::hypot(l.unit_vector[0],l.unit_vector[1]);
}
#ifndef LEGACY_BASELINE
void indexTests() {
  std::vector<Pt2D> q; std::vector<unsigned char> valid;
  for(int i=-3;i<70;++i) for(int j=-3;j<55;++j) {
    q.emplace_back(i*16.+(j%3)*0.8,j*16.); valid.push_back(1);
    if((i+j)%19==0) { q.emplace_back(NAN,0); valid.push_back(0); }
  }
  // Strip and exact-cell boundaries, including original IDs after invalids.
  for(Pt2D p:std::vector<Pt2D>{{-4,-4},{-0.8,0},{0.8,0},{1004,804},{1100,-8}}) {
    q.push_back(p); valid.push_back(1);
  }
  auto compare=[&](const std::vector<Pt2D>& pixels,const std::vector<unsigned char>& flags,
                   const std::string& name) {
    IDMap map(800,1000,4); map.rebuildCoordinates(pixels,flags);
    std::vector<Line2D> lines{{{0,0},{1,0}},{{0,0},{0,1}},{{-4,-4},{1,1}},
                              {{1004,804},{-2,1}},{{4,4},{1,-2}}};
    int comparisons=0;
    for(size_t a=0;a<lines.size();++a) for(size_t b=a;b<lines.size();++b) {
      std::vector<Line2D> pair{lines[a],lines[b]};
      std::vector<IDMap::RowSpan> spans; std::vector<int> actual,expected;
      map.computeStripIntersection(pair,0.8,spans);
      map.visitPointsInRowSpans(spans,pair,0.8,actual);
      for(int i=0;i<(int)pixels.size();++i)
        if(flags[i] && distance(pixels[i],pair[0])<=0.8 && distance(pixels[i],pair[1])<=0.8) expected.push_back(i);
      std::sort(actual.begin(),actual.end()); expect(actual==expected,name+" indexed/brute-force original IDs"); ++comparisons;
    }
    std::cout<<"INDEX "<<name<<" points="<<pixels.size()<<" comparisons="<<comparisons<<" scan="<<map.scansAll()<<'\n';
    std::vector<IDMap::RowSpan> spans; std::vector<int> ids;
    map.computeStripIntersection({{{NAN,0},{1,0}}},0.8,spans);
    map.visitPointsInRowSpans(spans,{{{NAN,0},{1,0}}},0.8,ids);
    expect(ids.empty(),"nonfinite line rejects without unsafe conversion");
  };
  compare(q,valid,"origin-aware");
  q.push_back({1e100,0}); valid.push_back(1);
  q.push_back({-1e100,0}); valid.push_back(1);
  compare(q,valid,"extreme-bounded-fallback");
  // Public raw rebuild owns its pointer vector (bindings pass a temporary).
  auto object=std::make_unique<Tracer2D>(Pt2D{16,16}); IDMap raw(100,100,4);
  raw.rebuild(std::vector<const Object2D*>{object.get()});
  std::vector<IDMap::RowSpan> spans; std::vector<int> ids;
  raw.computeStripIntersection({{{0,16},{1,0}}},0.8,spans);
  raw.visitPointsInRowSpans(spans,{{{0,16},{1,0}}},0.8,ids);
  expect(ids==std::vector<int>{0},"raw rebuild temporary-vector lifetime");
}

// Builds a NEW matcher for every changed calibration/observation snapshot.
// Returns matched true-world IDs; compares original measured pixels by their
// original (possibly interspersed) indices, never a compacted index.
std::set<int> fixture(std::vector<Pose> rig, std::vector<Pt3D> X, const Coeff& d,
    int build, bool invalid, const std::string& name, bool reverse=false,
    bool inactive=false, bool file_loaded=false) {
  std::vector<std::shared_ptr<Camera>> cams; Obs obs(rig.size());
  std::vector<std::vector<int>> ids(rig.size());
  for(int c=0;c<(int)rig.size();++c) {
    if(file_loaded) {
      // Setters deliberately retain their existing positive-focal contract.
      // Exercise signed calibration through the supported file-loading path.
      Pose positive=rig[c]; positive.fx=std::abs(positive.fx); positive.fy=std::abs(positive.fy);
      auto saved=camera(positive,d);
      saved->param().cam_mtx(0,0)=rig[c].fx; saved->param().cam_mtx(1,1)=rig[c].fy;
      std::string path="signed_"+std::to_string(c)+".txt";
      expect(bool(saved->saveParameters(path)),name+" save signed calibration");
      auto loaded=CameraFactory::loadFromFile(path);
      expect(bool(loaded),name+" load signed calibration");
      if(!loaded) return {};
      cams.push_back(loaded.value());
    } else cams.push_back(camera(rig[c],d));
    for(int i=0;i<(int)X.size();++i) {
      if(invalid) {
        // Nonfinite and finite fallback/nonconvergence in both reference and targets.
        obs[c].push_back(std::make_unique<Tracer2D>(Pt2D{NAN,4}));
        obs[c].push_back(std::make_unique<Tracer2D>(Pt2D{1e8,1e8}));
        obs[c].push_back(std::make_unique<Tracer2D>(Pt2D{INFINITY,4}));
      }
      ids[c].push_back((int)obs[c].size());
      obs[c].push_back(std::make_unique<Tracer2D>(oracle(rig[c],X[i],d)));
    }
    if(reverse) {
      std::reverse(obs[c].begin(),obs[c].end());
      for(auto& id:ids[c]) id=(int)obs[c].size()-1-id;
    }
    obs[c].push_back(std::make_unique<Tracer2D>(Pt2D{77.+c*41,101.+c*17}));
  }
  if(inactive) {
    auto poly=std::make_shared<PolynomialCamera>(); poly->is_active=false;
    cams.push_back(poly); obs.emplace_back();
  }
  auto cfg=config(build); StereoMatch matcher(cams,obs,cfg);
  expect(DebugAccess_StereoMatch::mode(matcher),name+" all-active-pinhole routing");
  for(int c=0;c<(int)rig.size();++c) for(int i=0;i<(int)X.size();++i) {
    auto q=cams[c]->project(X[i]); auto ray=cams[c]->lineOfSight(obs[c][ids[c][i]]->_pt_center);
    expect(q && (q.value()-oracle(rig[c],X[i],d)).norm()<1e-6,name+" independent projection");
    auto truth=X[i]-rig[c].center; truth=truth/truth.norm();
    expect(ray && myMATH::cross(truth,ray.value().unit_vector).norm()<1e-7,name+" independent ray");
    if(file_loaded) {
      auto pin=std::static_pointer_cast<PinholeCamera>(cams[c]); Pt2D pixel;
      expect(pin->matchingPixel(obs[c][ids[c][i]]->_pt_center,pixel),name+" signed checked inverse");
      expect((pixel-oracle(rig[c],X[i],Coeff(5,0))).norm()<1e-6,name+" signed matching pixels");
      // Independently forward-distort the recovered normalized coordinates.
      Pose normalized{0,{0,0,0},1,1,0,0};
      Pt3D recovered{(pixel[0]-rig[c].cx)/rig[c].fx,(pixel[1]-rig[c].cy)/rig[c].fy,1};
      auto distorted=oracle(normalized,recovered,d);
      Pt2D redistorted{rig[c].fx*distorted[0]+rig[c].cx,rig[c].fy*distorted[1]+rig[c].cy};
      expect((redistorted-obs[c][ids[c][i]]->_pt_center).norm()<=1e-6,name+" independent redistortion residual");
    }
  }
  if(invalid) {
    const auto& flags=DebugAccess_StereoMatch::valid(matcher);
    for(int c=0;c<(int)rig.size();++c) for(int id=0;id<(int)obs[c].size()-1;++id) {
      bool istrue=std::find(ids[c].begin(),ids[c].end(),id)!=ids[c].end();
      expect(bool(flags[c][id])==istrue,name+" individual inverse validity");
    }
  }
  // Query every ordered camera pair: candidate set must equal independent
  // undistorted-line brute-force selection PLUS reciprocal measured-ray test.
  for(int a=0;a<(int)rig.size();++a) for(int b=0;b<(int)rig.size();++b) if(a!=b) {
    int id=ids[a][0]; auto ray=cams[a]->lineOfSight(obs[a][id]->_pt_center);
    expect(bool(ray),name+" reference valid ray"); if(!ray) continue;
    std::vector<int> candidates;
    DebugAccess_StereoMatch::enumerate(matcher,{ray.value()},b,{a},{obs[a][id]->_pt_center},candidates);
    expect(std::find(candidates.begin(),candidates.end(),ids[b][0])!=candidates.end(),name+" reciprocal original target ID");
    Line2D line; expect(DebugAccess_StereoMatch::line(matcher,b,ray.value(),line),name+" analytic pair line");
    std::vector<int> brute;
    for(int j=0;j<(int)obs[b].size();++j) {
      if(!DebugAccess_StereoMatch::valid(matcher)[b][j]) continue;
      auto back=cams[b]->lineOfSight(obs[b][j]->_pt_center); Line2D reciprocal;
      if(!back || !DebugAccess_StereoMatch::line(matcher,a,back.value(),reciprocal)) continue;
      const auto& pixels=DebugAccess_StereoMatch::pixels(matcher);
      if(distance(pixels[b][j],line)<=0.8 && distance(pixels[a][id],reciprocal)<=0.8) brute.push_back(j);
    }
    std::sort(candidates.begin(),candidates.end()); expect(candidates==brute,name+" pair candidate set brute-force");
  }
  double max3d=0;
  auto identify=[&](const std::vector<std::unique_ptr<Object3D>>& result) {
    std::set<int> found;
    for(auto& r:result) {
      int k=-1; double err=1e100;
      for(int i=0;i<(int)X.size();++i) if((r->_pt_center-X[i]).norm()<err) {err=(r->_pt_center-X[i]).norm(); k=i;}
      expect(err<=1e-5*15,name+" known 3D fidelity");
      max3d=std::max(max3d,err);
      expect(found.insert(k).second,name+" no ghosts/duplicates");
      for(int c=0;c<(int)rig.size();++c) if(r->_obj2d_list[c])
        expect((r->_obj2d_list[c]->_pt_center-obs[c][ids[c][k]]->_pt_center).norm()==0,name+" measured pixels and uncompressed IDs");
    }
    expect(result.size()==X.size(),name+" 100% exact recall"); return found;
  };
  omp_set_num_threads(1); auto one=matcher.match(); auto found=identify(one);
  omp_set_num_threads(4); auto multi=matcher.match(); auto found_multi=identify(multi);
  expect(found==found_multi,name+" single/multithread sets");
  // Compare the COMPLETE cloned-observation signature too, not merely counts.
  auto signature=[](const std::vector<std::unique_ptr<Object3D>>& rs) {
    std::set<std::vector<double>> s;
    for(auto& r:rs) { std::vector<double> v; for(auto& o:r->_obj2d_list)
      {v.push_back(o?o->_pt_center[0]:-1e200); v.push_back(o?o->_pt_center[1]:-1e200);} s.insert(v); } return s;
  };
  expect(signature(one)==signature(multi),name+" complete correspondence sets across threads");
  for(int c=0;c<(int)rig.size();++c) for(int i=0;i<(int)X.size();++i)
    expect((obs[c][ids[c][i]]->_pt_center-oracle(rig[c],X[i],d)).norm()==0,
           name+" source measured pixels never mutated");
  omp_set_num_threads(1);
  std::cout<<"FIXTURE "<<name<<" points="<<X.size()<<" matches="<<one.size()<<" invalid="<<invalid<<" build="<<build<<" max3d="<<max3d<<'\n';
  return found;
}

void signedTests() {
  for(int signs=0;signs<3;++signs) for(bool distorted:{false,true}) for(int build:{2,3,4}) {
    std::vector<Pose> rig(poses.begin(),poses.end());
    for(auto& p:rig) { if(signs!=0) p.fx=-p.fx; if(signs!=1) p.fy=-p.fy; }
    const Coeff d=distorted?Coeff{-0.08,0.02,0.0004,-0.0004,-0.001}:Coeff(5,0);
    try { fixture(rig,points(),d,build,false,"signed_file_"+std::to_string(signs)+"_"+
                  std::to_string(distorted)+"_"+std::to_string(build),false,false,true); }
    catch(const std::exception& e) { expect(false,std::string("signed file calibration rejected: ")+e.what()); }
  }
  for(int axis:{0,1}) for(double value:std::array<double,5>{0.,-0.,NAN,INFINITY,-INFINITY}) {
    auto cam=camera(poses[0],Coeff(5,0)); cam->param().cam_mtx(axis,axis)=value;
    Pt2D pixel; expect(!cam->matchingPixel({483,397},pixel),"invalid focal checked conversion rejects");
    std::vector<std::shared_ptr<Camera>> cams{cam,camera(poses[1],Coeff(5,0))};
    Obs obs(2); auto cfg=config(2); bool caught=false;
    try { StereoMatch matcher(cams,obs,cfg); } catch(const std::exception&) { caught=true; }
    expect(caught,"zero/nonfinite focal constructor rejects");
  }
}

void inverseTests() {
  auto cam=camera(poses[0],{-0.2,0.05,0.001,-0.001,-0.01}); Pt2D q;
  expect(!cam->matchingPixel({1e8,1e8},q),"finite inverse fallback rejected");
  expect(!cam->matchingPixel({NAN,2},q),"NaN inverse rejected");
  expect(!cam->matchingPixel({2,INFINITY},q),"infinite inverse rejected");
  auto positive=camera(poses[0],{0.08,0,0,0,0});
  expect(!positive->matchingPixel({1e8,1e8},q),"finite unconverged positive-radial inverse rejected");
  // Valid fixtures must satisfy the stricter boundary threshold independently.
  for(auto& X:points()) {
    auto measured=oracle(poses[0],X,{-0.2,0.05,0.001,-0.001,-0.01});
    expect(cam->matchingPixel(measured,q),"checked inverse positive fixture");
    auto ideal=oracle(poses[0],X,Coeff(5,0));
    expect((ideal-q).norm()<1e-6,"checked undistorted pixel independent truth");
  }
  Obs obs(2); std::vector<std::shared_ptr<Camera>> cams{cam,camera(poses[1],Coeff(5,0))}; auto cfg=config(2);
  cam->param().cam_mtx(0,0)=NAN; bool caught=false;
  try { StereoMatch bad(cams,obs,cfg); } catch(const std::exception&) {caught=true;}
  expect(caught,"structural nonfinite calibration fails construction");
  cam=camera(poses[0],Coeff(5,0)); cams[0]=cam; cam->param().r_mtx=Matrix<double>(3,3,0.); caught=false;
  try { StereoMatch bad(cams,obs,cfg); } catch(const std::exception&) {caught=true;}
  expect(caught,"structural singular pose fails construction");
}

void lineTests() {
  const Coeff d{-0.08,0.02,0.0004,-0.0004,-0.001};
  for(double scale:{1e-6,1.,1e6}) {
    auto rig=poses; auto X=points()[0]*scale;
    for(auto& p:rig) p.center=p.center*scale;
    std::vector<std::shared_ptr<Camera>> cams{camera(rig[0],d),camera(rig[1],d)};
    Obs obs(2); auto cfg=config(2); StereoMatch m(cams,obs,cfg);
    auto v=X-rig[0].center; v=v/v.norm();
    Line2D first; Line3D ray{rig[0].center,v};
    expect(DebugAccess_StereoMatch::line(m,1,ray,first),"analytic scaled LOS");
    auto ideal=oracle(rig[1],X,Coeff(5,0));
    expect(distance(ideal,first)<1e-6,"world-unit invariance true epipolar point");
    for(double t:{-100.,0.,100.}) {
      Line2D line; Line3D moved{ray.pt+v*(t*scale),v*7};
      expect(DebugAccess_StereoMatch::line(m,1,moved,line),"LOS origin/direction translation status");
      expect(distance(ideal,line)<1e-6,"LOS origin-translation invariance");
      expect(std::abs(line.unit_vector[0]*first.unit_vector[1]-line.unit_vector[1]*first.unit_vector[0])<1e-10,
             "invariant analytic line direction");
    }
    Line2D unused;
    expect(!DebugAccess_StereoMatch::line(m,1,{rig[1].center,v},unused),"line through target center rejects");
    expect(!DebugAccess_StereoMatch::line(m,1,{rig[0].center,{0,0,0}},unused),"zero LOS direction rejects");
    expect(!DebugAccess_StereoMatch::line(m,1,{rig[0].center,{NAN,0,1}},unused),"nonfinite LOS rejects");
    const auto& target=std::static_pointer_cast<PinholeCamera>(cams[1])->param();
    expect(!DebugAccess_StereoMatch::line(m,1,
        {target.t_vec_inv+target.r_mtx_inv*Pt3D{scale,0,0},target.r_mtx_inv*Pt3D{0,1,0}},unused),
        "zero homogeneous image normal rejects");
    // Legacy exposed helper remains exactly its two measured projections.
    Line2D raw; expect(DebugAccess_StereoMatch::raw(m,1,ray,raw),"raw legacy helper status");
    auto a=cams[1]->project(ray.pt+v),b=cams[1]->project(ray.pt+v*2.);
    if(a && b) { expect((raw.pt-a.value()).norm()==0,"raw helper measured anchor unchanged");
      expect(distance(b.value(),raw)<1e-6,"raw helper measured second projection unchanged"); }
  }
  // Coincident centers: reject affected pair branch, not the whole match call.
  std::vector<std::shared_ptr<Camera>> cams{camera(poses[0],d),camera(poses[0],d)};
  Obs obs(2); for(auto& o:obs) o.push_back(std::make_unique<Tracer2D>(oracle(poses[0],points()[0],d)));
  auto cfg=config(2); StereoMatch m(cams,obs,cfg);
  expect(m.match().empty(),"zero-baseline pair rejects without throwing");
}

void snapshotTests() {
  std::vector<Pose> rig(poses.begin(),poses.end()); Coeff d{-0.08,0.02,0.0004,-0.0004,-0.001};
  auto X=points(); std::vector<std::shared_ptr<Camera>> cams;
  for(auto& p:rig) cams.push_back(camera(p,d));
  Obs obs(4); auto cfg=config(3);
  auto refill=[&]() {
    for(int c=0;c<4;++c) { obs[c].clear(); for(auto& x:X)
      obs[c].push_back(std::make_unique<Tracer2D>(oracle(rig[c],x,d))); }
  };
  auto check=[&](const std::string& name) {
    StereoMatch matcher(cams,obs,cfg); auto result=matcher.match();
    expect(result.size()==X.size(),name+" fresh matcher recall"); std::set<int> identities;
    for(auto& r:result) {
      int best=-1; double err=1e100;
      for(int i=0;i<(int)X.size();++i) if((r->_pt_center-X[i]).norm()<err) {err=(r->_pt_center-X[i]).norm(); best=i;}
      expect(err<15e-5,name+" fresh matcher ground truth"); identities.insert(best);
      for(int c=0;c<4;++c) if(r->_obj2d_list[c])
        expect((r->_obj2d_list[c]->_pt_center-oracle(rig[c],X[best],d)).norm()==0,
               name+" cloned current pixels/IDs");
    }
    expect(identities.size()==X.size(),name+" exact physical identity set");
  }; // matcher dies BEFORE every following mutation of the reused inputs
  refill(); check("snapshot initial");
  rig[0].fx+=25; rig[0].cx+=10;
  std::static_pointer_cast<PinholeCamera>(cams[0])->setIntrinsics(rig[0].fx,rig[0].fy,rig[0].cx,rig[0].cy,d);
  rig[1].angle+=0.02; rig[1].center[0]+=0.1; cams[1]=camera(rig[1],d);
  X[0][0]+=0.17; X.push_back({0.8,1.0,12.9}); refill();
  for(auto& o:obs) std::reverse(o.begin(),o.end());
  check("snapshot reused mutated intrinsics-pose-observation-size-order");
  cams[3]->is_active=false; cfg._sm_param.match_cam_count=2;
  check("snapshot reused active flag and config");
  cams[3]->is_active=true; std::swap(cams[0],cams[2]); std::swap(obs[0],obs[2]); std::swap(rig[0],rig[2]);
  check("snapshot reused camera ordering");
}

void referenceCountTest() {
  Coeff d{-0.2,0.05,0.001,-0.001,-0.01}; Obs obs(4);
  std::vector<std::shared_ptr<Camera>> cams; auto X=points();
  for(int c=0;c<4;++c) {
    cams.push_back(camera(poses[c],d));
    for(auto& x:X) obs[c].push_back(std::make_unique<Tracer2D>(oracle(poses[c],x,d)));
    if(c==2) for(int k=0;k<50;++k) obs[c].push_back(std::make_unique<Tracer2D>(Pt2D{NAN,4}));
    else obs[c].push_back(std::make_unique<Tracer2D>(Pt2D{50.+c*20,80.+c*31}));
  }
  auto cfg=config(2); StereoMatch matcher(cams,obs,cfg); auto matches=matcher.match();
  expect(matches.size()==X.size(),"valid-count reference scene recall");
  for(auto& m:matches) expect(bool(m->_obj2d_list[2]),"reference selected by valid count, not raw count");
}
#endif

void mixedTests() {
  // A uniform-index stack is an independently known straight optical path;
  // the polynomial fixture is an exactly invertible affine orthographic map.
  for(bool refractive:{false,true}) for(bool swap:{false,true}) for(bool distorted:{false,true}) {
    Pose pin=poses[0]; Coeff d=distorted?Coeff{-0.08,0.02,0.0004,-0.0004,-0.001}:Coeff(5,0);
    std::shared_ptr<Camera> other;
    if(refractive) {
      auto ref=std::make_shared<RefractionPinholeCamera>(); Pose p=poses[1];
      auto central=camera(p,Coeff(5,0));
      static_cast<PinholeParam&>(ref->param())=central->param();
      ref->setRefraction({0,0,3},{0,0,1},{1,1,1},{0.5});
      ref->setSolverOptions(1e-10,100,0.1); ref->commitUpdate(); other=ref;
    } else {
      auto poly=std::make_shared<PolynomialCamera>(); poly->setImageSize(800,1000);
      poly->setReferencePlane(REF_Z,5,15);
      poly->setCoefficients(Matrix<double>{{500,0,0,0},{40,1,0,0},{0,0,1,0}},
                            Matrix<double>{{400,0,0,0},{0,1,0,0},{40,0,1,0}});
      poly->commitUpdate(); other=poly;
    }
    std::vector<std::shared_ptr<Camera>> cams{camera(pin,d),other};
    if(swap) std::swap(cams[0],cams[1]);
    Obs obs(2); const auto X=points(); int pinid=swap?1:0;
    for(auto& x:X) {
      auto qpin=oracle(pin,x,d);
      Pt2D qother=refractive?oracle(poses[1],x,Coeff(5,0)):Pt2D{500+40*x[0],400+40*x[1]};
      obs[pinid].push_back(std::make_unique<Tracer2D>(qpin));
      obs[1-pinid].push_back(std::make_unique<Tracer2D>(qother));
    }
    auto cfg=config(2); StereoMatch m(cams,obs,cfg);
    expect(!DebugAccess_StereoMatch::mode(m),"mixed uses complete raw legacy route");
    for(int c=0;c<2;++c) {
      auto ray=cams[c]->lineOfSight(obs[c][0]->_pt_center); Line2D match,raw;
      expect(ray && DebugAccess_StereoMatch::line(m,1-c,ray.value(),match) &&
             DebugAccess_StereoMatch::raw(m,1-c,ray.value(),raw),"mixed pair helpers valid");
      expect((match.pt-raw.pt).norm()==0 && (match.unit_vector-raw.unit_vector).norm()==0,
             "mixed matching helper identical to raw approximation");
    }
    auto result=m.match(); std::set<int> found;
    for(auto& r:result) {
      int k=-1; double best=1e100;
      for(int i=0;i<(int)X.size();++i) if((r->_pt_center-X[i]).norm()<best) {best=(r->_pt_center-X[i]).norm(); k=i;}
      expect(best<=15e-5,"mixed recovered known scene fidelity"); found.insert(k);
      for(int c=0;c<2;++c)
        expect(r->_obj2d_list[c] && (r->_obj2d_list[c]->_pt_center-obs[c][k]->_pt_center).norm()==0,
               "mixed original measured pixels unchanged");
    }
    // Frozen HEAD-source executable established these exact identities.
    // The refractive +1/+2 construction is already degenerate for this stack;
    // preserving zero recovery is compatibility evidence, NOT physics success.
    std::set<int> expected;
    if(!refractive) { expected={0,1,2,3,4,5}; if(distorted && !swap) expected.erase(4); }
    expect(found==expected,"mixed exact legacy baseline correspondence identities");
    std::cout<<"MIXED "<<(refractive?"refractive":"polynomial")<<" swap="<<swap<<" distorted="<<distorted
             <<" matches="<<result.size()<<" IDs="; for(int k:found) std::cout<<k<<','; std::cout<<'\n';
  }
}
}
int main(int argc,char **argv) {
  auto start=std::chrono::steady_clock::now(); omp_set_num_threads(1);
  bool smoke=argc>1 && std::string(argv[1])=="--smoke";
  bool legacy=argc>1 && std::string(argv[1])=="--legacy-baseline";
  if(legacy) { scene(Coeff(5,0),4,"zero"); mixedTests();
    std::cout<<"CHECKS "<<checks<<" FAILURES "<<failures<<'\n'; return failures?1:0; }
#ifndef LEGACY_BASELINE
  if(argc>1 && std::string(argv[1])=="--signed") {
    signedTests(); std::cout<<"CHECKS "<<checks<<" FAILURES "<<failures<<'\n'; return failures?1:0;
  }
  signedTests();
  loader(); scene(Coeff(5,0),4,"zero");
  scene({-0.08,0.02,0.0004,-0.0004,-0.001},2,"combined");
  inverseTests();
  if(!smoke) {
    for(int i=0;i<5;++i) for(int sign:{-1,1}) {
      Coeff d(5,0); d[i]=sign*std::array<double,5>{0.08,0.02,0.0004,0.0004,0.001}[i];
      scene(d,3,"coefficient_"+std::to_string(i)+"_"+std::to_string(sign));
    }
    scene({-0.2,0.05,0.001,-0.001,-0.01},4,"stress");
    scene({0.2,-0.05,-0.001,0.001,0.01},4,"stress_sign_variant");
    scene({-0.08,0.02,0.0004,-0.0004,-0.001,0.001,-0.0002,0.0001},3,"rational8");
    scene({-0.08,0.02,0.0004,-0.0004,-0.001,0.001,-0.0002,0.0001,0.0001,-0.0001,0.0002,-0.0001},3,"prism12");
    for(int i:{1,2,3}) for(int sign:{-1,1}) {
      Coeff d(5,0); d[i]=sign*(i==1?1e-4:1e-5);
      scene(d,2,"supplied_tiny_"+std::to_string(i)+"_"+std::to_string(sign));
    }
    indexTests(); lineTests(); snapshotTests(); referenceCountTest();
    mixedTests();
    std::vector<Pose> rig(poses.begin(),poses.end()); const Coeff stress{-0.2,0.05,0.001,-0.001,-0.01};
    auto valid=fixture(rig,points(),stress,2,false,"valid-only");
    for(int order=0;order<4;++order) {
      auto invalid=fixture(rig,points(),stress,order%2?3:2,true,"invalid_order_"+std::to_string(order));
      expect(invalid==valid,"invalid skip matches valid-only physical identities");
      std::rotate(rig.begin(),rig.begin()+1,rig.end());
    }
    fixture(rig,points(),stress,4,false,"inactive-polynomial",false,true);
    // Fresh snapshots after intrinsics, pose, active set, and observation changes.
    fixture(rig,points(),stress,3,false,"before-mutations");
    rig[0].fx*=1.03; rig[1].cx+=17; rig[2].angle+=0.02; rig[3].center[0]+=0.1;
    auto changed=points(); changed[0][0]+=0.2; changed.push_back({0.9,1.1,12.4});
    fixture(rig,changed,stress,3,false,"fresh-calibration-observation-size-order",true);
    rig.pop_back(); fixture(rig,changed,stress,2,false,"fresh-active-set",true);
    std::mt19937 random(317); std::uniform_real_distribution<double> xy(-1.4,1.4),z(9.5,14.);
    std::vector<Pt3D> seeded; for(int i=0;i<20;++i) seeded.push_back({xy(random),xy(random),z(random)});
    fixture(rig,seeded,stress,3,false,"seeded20");
    std::vector<Pose> edge{{0,{-0.4,0,0},500,500,500,400},{0,{0.4,0,0},500,500,500,400},
                           {0,{0,0.3,0},500,500,500,400},{0,{0,-0.3,0},500,500,500,400}};
    std::vector<Pt3D> edgeX{{-8.2,0.7,8},{8.2,-0.6,8},{0.5,-6.4,8},{-0.7,6.4,8}};
    for(auto& p:edge) for(auto& x:edgeX) {
      auto measured=oracle(p,x,{-0.08,0,0,0,0});
      expect(measured[0]>=0 && measured[0]<1000 && measured[1]>=0 && measured[1]<800,
             "edge fixture measured pixels inside sensor");
    }
    fixture(edge,edgeX,{-0.08,0,0,0,0},2,false,"undistorted-outside-sensor");
  }
#endif
  std::cout<<"CHECKS "<<checks<<" FAILURES "<<failures<<" seconds="
           <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
  return failures?1:0;
}
