#include "myMATH.h"
#include <functional>

namespace myMATH
{

// Linespace
// n >= 2, including min and max, n_gap = n-1
std::vector<double> linspace (double min, double max, int n)
{
    // both min and max are stored in the vector
    if (n < 2)
    {
        std::cerr << "myMATH::Linspace: Cannot generate line space: "
                  << "n = " << n << ", "
                  << "less than 2!"
                  << std::endl;
        throw error_size;
    }

    double delta = (max - min) / (n - 1);

    std::vector<double> res(n);
    for (int i = 0; i < n; i ++)
    {
        res[i] = (min + delta * i);
    }

    return res;
}

//
// Generate all size-K combinations from the set {0, 1, ..., N-1}.
// - Output is in lexicographic order (e.g. N=4, K=3 -> [0,1,2], [0,1,3], [0,2,3], [1,2,3]).
// - Uses a short DFS (backtracking) with two key invariants:
//   1) We grow `comb` from left to right; at depth `d` we choose the next index i >= `start`.
//   2) Prune when there are not enough elements left: i can go up to N - (K - d).
//      (Because we still need (K - d) slots to fill after choosing position d.)
void generateCombinations(size_t N, size_t K, std::vector<std::vector<int>>& out)
{
    out.clear();
    if (K > N) return;
    if (K == 0) { out.push_back({}); return; }

    std::vector<int> comb;
    comb.reserve(K);

    // DFS(start, depth): this is a definition of the recursive function
    // - `start`: the minimum candidate value for the next position
    // - `depth`: how many elements are already placed in `comb`
    std::function<void(size_t, size_t)> DFS = [&](size_t start, size_t depth)
    {
        if (depth == K) {
            // We placed K elements, record one combination
            out.push_back(comb);
            return;
        }

        // Choose value for comb[depth]. The maximum `i` we can pick is:
        // N - (K - depth), so that there will still be enough numbers left to fill.
        for (size_t i = start; i <= N - (K - depth); ++i) {
            comb.push_back(static_cast<int>(i));   // place current choice
            DFS(i + 1, depth + 1);                 // recurse with next start and depth
                                                   // this step will recursively fill comb until depth == K
            comb.pop_back();                       // after one combination is done, remove last element and find
                                                   // the next candidate for comb[depth]
        }
    };

    DFS(0, 0); // Start DFS with initial start=0 and depth=0
}

// Bilinear interpolation
double bilinearInterp(AxisLimit const& grid_limit, std::vector<double> const& value, std::vector<double> const& pt_vec)
{
    // [https://en.wikipedia.org/wiki/Bilinear_interpolation]
    // value: 4
    //  c_00 [x_min, y_min]
    //  c_10 [x_max, y_min]
    //  c_11 [x_max, y_max]
    //  c_01 [x_min, y_max]
    // pt_vec: 2
    //  x, y
    // Key requirement:
    // If pt_vec is outside the limit:
    //  set it as the value on the boundary 

    double x_0 = grid_limit.x_min;
    double x_1 = grid_limit.x_max;
    double y_0 = grid_limit.y_min;
    double y_1 = grid_limit.y_max;

    double c_00 = value[0];
    double c_10 = value[1];
    double c_11 = value[2];
    double c_01 = value[3];

    double x = pt_vec[0];
    double y = pt_vec[1];
    double x_d = (x - x_0) / (x_1 - x_0);
    double y_d = (y - y_0) / (y_1 - y_0);
    if (x_d > 1+SMALLNUMBER || x_d < -SMALLNUMBER || 
        y_d > 1+SMALLNUMBER || y_d < -SMALLNUMBER)
    {
        std::cerr << "myMATH::BilinearInterp error: out of range" 
                  << "(x,y) = (" 
                  << x << ","
                  << y << ")"
                  << "(xd,yd) = ("
                  << x_d << ","
                  << y_d << ")"
                  << "(x0,y0) = ("
                  << x_0 << ","
                  << y_0 << ")"
                  << "(x1,y1) = ("
                  << x_1 << ","
                  << y_1 << ")"
                  << std::endl;
        throw error_range;
    }
    x_d = std::max(0.0, std::min(1.0, x_d));
    y_d = std::max(0.0, std::min(1.0, y_d));

    double c_x0 = c_00 * (1 - x_d) + c_10 * x_d;
    double c_x1 = c_01 * (1 - x_d) + c_11 * x_d;
    double c = c_x0 * (1 - y_d) + c_x1 * y_d;
    return c;
}

// Trilinear interpolation
double triLinearInterp(AxisLimit const& grid_limit, std::vector<double> const& value, std::vector<double> const& pt_vec)
{
    // [https://en.wikipedia.org/wiki/Trilinear_interpolation]
    // value: 8
    //        c_000 [x_min, y_min, z_min]
    //        c_100 [x_max, y_min, z_min]
    //        c_101 [x_max, y_min, z_max]
    //        c_001 [x_min, y_min, z_max]
    //        c_010 [x_min, y_max, z_min]
    //        c_110 [x_max, y_max, z_min]
    //        c_111 [x_max, y_max, z_max]
    //        c_011 [x_min, y_max, z_max]
    // pt_world: 3 
    //           x, y, z
    
    // Key requirement:
    // If pt_world is outside the limit:
    //  set it as the value on the boundary 

    double x_0 = grid_limit.x_min;
    double x_1 = grid_limit.x_max;
    double y_0 = grid_limit.y_min;
    double y_1 = grid_limit.y_max;
    double z_0 = grid_limit.z_min;
    double z_1 = grid_limit.z_max;

    double c_000 = value[0];
    double c_100 = value[1];
    double c_101 = value[2];
    double c_001 = value[3];
    double c_010 = value[4];
    double c_110 = value[5];
    double c_111 = value[6];
    double c_011 = value[7];

    double x = pt_vec[0];
    double y = pt_vec[1];
    double z = pt_vec[2];

    double x_d = (x - x_0) / (x_1 - x_0);
    double y_d = (y - y_0) / (y_1 - y_0);
    double z_d = (z - z_0) / (z_1 - z_0);

    if (x_d > 1+SMALLNUMBER || x_d < -SMALLNUMBER || 
        y_d > 1+SMALLNUMBER || y_d < -SMALLNUMBER || 
        z_d > 1+SMALLNUMBER || z_d < -SMALLNUMBER)
    {
        std::cerr << "myMATH::TriLinearInterp error: out of range" 
                  << "(x,y,z) = (" 
                  << x << ","
                  << y << ","
                  << z << ")"
                  << "(xd,yd,zd) = ("
                  << x_d << ","
                  << y_d << ","
                  << z_d << ")"
                  << "(x0,y0,z0) = ("
                  << x_0 << ","
                  << y_0 << ","
                  << z_0 << ")"
                  << "(x1,y1,z1) = ("
                  << x_1 << ","
                  << y_1 << ","
                  << z_1 << ")"
                  << std::endl;
        throw error_range;
    }

    x_d = std::max(0.0, std::min(1.0, x_d));
    y_d = std::max(0.0, std::min(1.0, y_d));
    z_d = std::max(0.0, std::min(1.0, z_d));

    double c_00 = c_000 * (1 - x_d) + c_100 * x_d;
    double c_01 = c_001 * (1 - x_d) + c_101 * x_d;
    double c_10 = c_010 * (1 - x_d) + c_110 * x_d;
    double c_11 = c_011 * (1 - x_d) + c_111 * x_d;

    double c_0 = c_00 * (1 - y_d) + c_10 * y_d;
    double c_1 = c_01 * (1 - y_d) + c_11 * y_d;

    double c = c_0 * (1 - z_d) + c_1 * z_d;

    return c;
}

// Create unit vector
// unit_vec = (pt2 - pt1) / norm(pt2 - pt1)
Pt3D createUnitVector (Pt3D const& pt1, Pt3D const& pt2)
{
    Pt3D res = pt2 - pt1;
    res /= res.norm();
    return res;
}

// Create unit vector
// unit_vec = (pt2 - pt1) / norm(pt2 - pt1)
Pt2D createUnitVector (Pt2D const& pt1, Pt2D const& pt2)
{
    Pt2D res = pt2 - pt1;
    res /= res.norm();
    return res;
}

// Calculate dot product
double dot (Pt3D const& pt1, Pt3D const& pt2)
{
    double res = 0;
    for (int i = 0; i < 3; i ++)
    {
        res += pt1[i] * pt2[i];
    }
    return res;
}

// Calculate dot product
double dot (Pt2D const& pt1, Pt2D const& pt2)
{
    double res = 0;
    for (int i = 0; i < 2; i ++)
    {
        res += pt1[i] * pt2[i];
    }
    return res;
}

// Calculate the distance between two points
double dist2 (Pt3D const& pt1, Pt3D const& pt2)
{
    double res = 0;
    for (int i = 0; i < 3; i ++)
    {
        res += std::pow(pt2[i] - pt1[i], 2);
    }

    res = std::max(res, 0.0);
    return res;
}

// Calculate the distance between two points
double dist2 (Pt2D const& pt1, Pt2D const& pt2)
{
    double res = 0;
    for (int i = 0; i < 2; i ++)
    {
        res += std::pow(pt2[i] - pt1[i], 2);
    }

    res = std::max(res, 0.0);
    return res;
}

// Calculate the distance between point and line
double dist2 (Pt3D const& pt, Line3D const& line)
{
    Pt3D diff = pt - line.pt;

    // double dist_proj = dot(diff, line.unit_vector);
    // double distance = dot(diff, diff) - std::pow(dist_proj, 2);
    // distance = std::max(distance, 0.0);

    double x = diff[1]*line.unit_vector[2] - diff[2]*line.unit_vector[1];
    double y = diff[2]*line.unit_vector[0] - diff[0]*line.unit_vector[2];
    double z = diff[0]*line.unit_vector[1] - diff[1]*line.unit_vector[0];
    double distance = x*x + y*y + z*z;
    
    return distance;
}

// Calculate the distance between point and line
double dist2 (Pt2D const& pt, Line2D const& line)
{
    Pt2D diff = pt - line.pt;

    // double dist_proj = dot(diff, line.unit_vector);
    // double distance = dot(diff, diff) - std::pow(dist_proj, 2);
    // distance = std::max(distance, 0.0);

    double z = diff[1]*line.unit_vector[0] - diff[0]*line.unit_vector[1];
    double distance = z*z;

    return distance;
}

// Calculate the distance between point and plane
double dist2 (Pt3D const& pt, Plane3D const& plane)
{
    Pt3D diff = pt - plane.pt;
    double distance = std::pow(dot(diff, plane.norm_vector), 2);
    distance = std::max(distance, 0.0);

    return distance;
}

// Calculate the distance between two points
double dist (Pt3D const& pt1, Pt3D const& pt2)
{
    return std::sqrt(dist2(pt1, pt2));
}

// Calculate the distance between two points
double dist (Pt2D const& pt1, Pt2D const& pt2)
{
    return std::sqrt(dist2(pt1, pt2));
}

// Calculate the distance between point and line
double dist (Pt3D const& pt, Line3D const& line)
{
    return std::sqrt(dist2(pt, line));
}

// Calculate the distance between point and line
double dist (Pt2D const& pt, Line2D const& line)
{
    return std::sqrt(dist2(pt, line));
}

// Calculate the distance between point and plane
double dist (Pt3D const& pt, Plane3D const& plane)
{
    Pt3D diff = pt - plane.pt;
    double distance = dot(diff, plane.norm_vector);
    return std::fabs(distance);
}

double pointPlaneSignedDistance(Pt3D const& pt, Plane3D const& plane)
{
    Pt3D diff = pt - plane.pt;
    return dot(diff, plane.norm_vector);
}

Pt3D projectPointToPlaneAlongNormal(Pt3D const& pt, Plane3D const& plane)
{
    double signed_dist = pointPlaneSignedDistance(pt, plane);
    return pt - plane.norm_vector * signed_dist;
}

// Triangulation
void triangulation(Pt3D& pt_world, double& error,
                   std::vector<Line3D> const& line_of_sight_list)
{
    int n = line_of_sight_list.size();
    REQUIRE(n >= 2, ErrorCode::InvalidArgument,"myMATH::Triangulation: "
                        "Cannot triangulate with less than 2 lines of sight!");

    Matrix<double> mtx(3, 3, 0);
    Matrix<double> temp(3,3,0);
    Pt3D pt_3d(0,0,0);
    Pt3D pt_ref;
    Pt3D unit_vector;

    for (int i = 0; i < n; i ++)
    {
        pt_ref = line_of_sight_list[i].pt;
        unit_vector = line_of_sight_list[i].unit_vector;

        double nx = unit_vector[0];
        double ny = unit_vector[1];
        double nz = unit_vector[2];

        temp(0,0) = 1-nx*nx; temp(0,1) = -nx*ny;  temp(0,2) = -nx*nz;
        temp(1,0) = -nx*ny;  temp(1,1) = 1-ny*ny; temp(1,2) = -ny*nz;
        temp(2,0) = -nx*nz;  temp(2,1) = -ny*nz;  temp(2,2) = 1-nz*nz;

        pt_3d += temp * pt_ref;
        mtx += temp;
    }

    pt_world = myMATH::inverse(mtx) * pt_3d;

    // calculate errors 
    error = 0.0;
    Pt3D diff_vec;
    for (int i = 0; i < n; i ++)
    {
        diff_vec = pt_world - line_of_sight_list[i].pt;
        double h = std::pow(diff_vec.norm(),2)
            - std::pow(myMATH::dot(diff_vec, line_of_sight_list[i].unit_vector),2);
            
        if (h < 0.0 && h >= - SMALLNUMBER)
        {
            h = 0.0;
        }
        else if (h < - SMALLNUMBER)
        {
            std::cerr << "myMATH::triangulation error: "
                      << "the distance in triangulation is: "
                      << "h = "
                      << h
                      << std::endl;
            throw error_range;
        }
        else
        {
            h = std::sqrt(h);
        }

        error += h * h;
    }
    error = std::sqrt(error / n);
}

// Cross product of two 3d vectors
Pt3D cross (Pt3D const& vec1, Pt3D const& vec2)
{
    Pt3D res;
    res[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    res[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    res[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
    return res;
}


// Find the cross points of two 2d lines
bool crossPoint (Pt2D& pt2d, Line2D const& line1, Line2D const& line2)
{
    double den = line1.unit_vector[0] * line2.unit_vector[1] - line1.unit_vector[1] * line2.unit_vector[0];
    
    // if (std::fabs(den) < SMALLNUMBER)
    // {
    //     std::cerr << "myMATH::crossPoint: "
    //               << "The two lines are parallel!"
    //               << std::endl;
    //     throw error_range;
    // }
    bool is_parallel = false;
    if (std::fabs(den) < 1e-10)
    {
        std::cout << "myMATH::crossPoint warning:"
                  << "The two lines are parallel!"
                  << std::endl;
        is_parallel = true;
        pt2d[0] = 0;
        pt2d[1] = 0;
        return is_parallel;
    }

    double num = line2.unit_vector[1] * (line2.pt[0] - line1.pt[0]) - line2.unit_vector[0] * (line2.pt[1] - line1.pt[1]);
    double factor = num / den;

    pt2d[0] = (line1.unit_vector[0]) * factor + line1.pt[0];
    pt2d[1] = (line1.unit_vector[1]) * factor + line1.pt[1];

    return is_parallel;
}

// Find the cross points of 3d line and 3d plane
bool crossPoint (Pt3D& pt3d, Line3D const& line, Plane3D const& plane)
{
    double den = myMATH::dot(line.unit_vector, plane.norm_vector);
    
    bool is_parallel = false;
    if (std::fabs(den) < 1e-10)
    {
        std::cout << "myMATH::crossPoint warning:"
                  << "The two lines are parallel!"
                  << std::endl;
        is_parallel = true;
        for (int i = 0; i < 3; i ++)
        {
            pt3d[i] = 0;
        }
        return is_parallel;
    }

    for (int i = 0; i < 3; i ++)
    {
        pt3d[i] = plane.pt[i] - line.pt[i];
    }

    double num = myMATH::dot(pt3d, plane.norm_vector);
    double factor = num / den;

    for (int i = 0; i < 3; i ++)
    {
        pt3d[i] = line.unit_vector[i] * factor + line.pt[i];
    }

    return is_parallel;
}

// Polynomial fit
void polyfit (std::vector<double>& coeff, std::vector<double> const& x, std::vector<double> const& y, int order)
{
    if (x.size() != y.size())
    {
        std::cerr << "myMATH::polyfit error at line " << __LINE__ << ":\n"
                  << "x.size() != y.size()"
                  << std::endl;
        throw error_size;
    }

    if (order < 1)
    {
        std::cerr << "myMATH::polyfit error at line " << __LINE__ << ":\n"
                  << "order < 1"
                  << std::endl;
        throw error_range;
    }


    int n = x.size();
    int m = order + 1;
    
    Matrix<double> x_mat(n, m, 0);
    Matrix<double> y_mat(n, 1, 0);
    Matrix<double> a_mat(m, 1, 0);

    for (int i = 0; i < n; i ++)
    {
        for (int j = 0; j < m; j ++)
        {
            x_mat(i,j) = std::pow(x[i], j);
        }
        y_mat(i,0) = y[i];
    }

    // std::string method = m < 4 ? "det" : "gauss";
    std::string method = "gauss";
    a_mat = myMATH::inverse(x_mat.transpose() * x_mat, method) * x_mat.transpose() * y_mat;

    coeff.resize(m);
    for (int i = 0; i < m; i ++)
    {
        coeff[i] = a_mat(i,0);
    }
}

// Create Gaussian kernel
std::vector<double> createGaussianKernel(int radius, double sigma) 
{
    int size = 2 * radius + 1;
    std::vector<double> kernel(size);
    double sum = 0.0;

    for (int i = 0; i < size; i++) 
    {
        int x = i - radius;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize the kernel
    for (int i = 0; i < size; i++) 
    {
        kernel[i] /= sum;
    }

    return kernel;
}

// img: image that need to perform crosscorrelation, ref_img: the reference image, (cx, cy): the location of pixel where the center of ref_img is
// NCC between an image patch and a reference image placed with its geometric center at (cx, cy).
// - Geometric center of ref: ((col_ref-1)/2, (row_ref-1)/2)
// - Nearest-neighbor sampling in ref via llround
// - Window in img is chosen so that mapped (dy,dx) are always in-bounds for ref_img
double imgCrossCorrAtPt(const Image& img, const Image& ref_img, double cx, double cy)
{
    const int row_img = img.getDimRow();
    const int col_img = img.getDimCol();
    const int row_ref = ref_img.getDimRow();
    const int col_ref = ref_img.getDimCol();

    // Geometric center of the reference
    const double col_ref_center = (col_ref - 1) * 0.5;
    const double row_ref_center = (row_ref - 1) * 0.5;

    // Build intersection window in IMG so that after rounding:
    //   dx = round(col_ref_center + (col - col_c)) ∈ [0, col_ref-1]
    //   dy = round(row_ref_center + (row - row_c)) ∈ [0, row_ref-1]
    // Since round(u) ∈ [0..N-1]  <=>  u ∈ [-0.5, N-0.5)
    const int col_min = std::max(0,       static_cast<int>(std::ceil (cx - (col_ref_center + 0.5))));
    const int col_max = std::min(col_img, static_cast<int>(std::floor(cx + (col_ref - 0.5 - col_ref_center)) + 1));
    const int row_min = std::max(0,       static_cast<int>(std::ceil (cy - (row_ref_center + 0.5))));
    const int row_max = std::min(row_img, static_cast<int>(std::floor(cy + (row_ref - 0.5 - row_ref_center)) + 1));

    if (col_min >= col_max || row_min >= row_max) return 0.0;  // no overlap

    // 1) Means
    double sum_img = 0.0, sum_ref = 0.0;
    long   n_sum   = 0;
    for (int r = row_min; r < row_max; ++r) {
        for (int c = col_min; c < col_max; ++c) {
            const int dx = static_cast<int>(std::llround(col_ref_center + (c - cx)));
            const int dy = static_cast<int>(std::llround(row_ref_center + (r - cy)));
            sum_img += img(r, c);
            sum_ref += ref_img(dy, dx);
            ++n_sum;
        }
    }
    if (n_sum == 0) return 0.0;

    const double mu_img = sum_img / static_cast<double>(n_sum);
    const double mu_ref = sum_ref / static_cast<double>(n_sum);

    // 2) Zero-mean NCC
    double num = 0.0, den_img = 0.0, den_ref = 0.0;
    for (int r = row_min; r < row_max; ++r) {
        for (int c = col_min; c < col_max; ++c) {
            const int dx = static_cast<int>(std::llround(col_ref_center + (c - cx)));
            const int dy = static_cast<int>(std::llround(row_ref_center + (r - cy)));
            const double ai = img(r, c)       - mu_img;
            const double bi = ref_img(dy, dx) - mu_ref;
            num     += ai * bi;
            den_img += ai * ai;
            den_ref += bi * bi;
        }
    }

    // 3) Normalize with guards
    constexpr double eps = 1e-12;
    double corr = 0.0;
    if (den_img > eps && den_ref > eps) {
        corr = num / std::sqrt(den_img * den_ref);
        if (corr >  1.0) corr =  1.0;
        if (corr < -1.0) corr = -1.0;
    } else if (den_img <= eps && den_ref <= eps) {
        corr = 1.0;  // both patches nearly constant
    } else {
        corr = 0.0;  // one constant, one not
    }

    return corr;
}


// Calculate image cross correlation
double imgCrossCorr(Image const& img, Image const& img_ref)
{
    int n_row = img.getDimRow();
    int n_col = img.getDimCol();
    int n = n_row * n_col;
    if (n == 0) {
        std::cerr << "myMATH::imgCrossCorr warning: "
                  << "Image dimensions are zero!" 
                  << std::endl;
        return -std::numeric_limits<double>::infinity();
    } 
    if (n_row != img_ref.getDimRow() || n_col != img_ref.getDimCol())
    {
        std::cerr << "myMATH::imgCrossCorr warning: "
                  << "Image dimensions do not match!" 
                  << std::endl;
        return -std::numeric_limits<double>::infinity();
    }

    double img_mean = 0;
    double img_ref_mean = 0;
    for (int i = 0; i < n; i++)
    {
        img_mean += img[i];
        img_ref_mean += img_ref[i];
    }
    img_mean /= n;
    img_ref_mean /= n;

    double val = 0;
    double img_var = 0;
    double img_ref_var = 0;
    for (int i = 0; i < n; i++)
    {
        img_var += (img[i] - img_mean) * (img[i] - img_mean);
        img_ref_var += (img_ref[i] - img_ref_mean) * (img_ref[i] - img_ref_mean);
        val += (img[i] - img_mean) * (img_ref[i] - img_ref_mean);
    }
    img_var = std::max(img_var, 0.0);
    img_ref_var = std::max(img_ref_var, 0.0);
    if (img_var > SMALLNUMBER && img_ref_var > SMALLNUMBER)
    {
        val /= std::sqrt(img_var * img_ref_var);
    }
    else if (img_var < SMALLNUMBER && img_ref_var < SMALLNUMBER)
    {
        val = 1.0; // both images are constant
    }
    else
    {
        val = 0.0;
    }

    // Clamp the value to the range [-1, 1]
    val = std::clamp(val, -1.0, 1.0);
    return val;
}

// Check whether an int belongs to a vector
bool ismember (int id, std::vector<int> const& vec)
{
    return std::find(vec.begin(), vec.end(), id) != vec.end();
}

// Compute the left-tail deletion threshold (low_cut) using KDE + modal HDR + guards.
// Returns a real low_cut only when the left side is truly sparse and separated from the main mode.
// Otherwise returns (min(x) - tiny), which effectively means "no deletion".
double computeLowCutKDE(const std::vector<double>& x_raw,
                               double p,           // modal HDR mass (candidate region width)
                               double ratio_thresh,// density ratio threshold f(L)/f(mode) (smaller = more conservative)
                               double valley_thresh)// valley depth threshold min_{(L,mode)} f / f(mode)
{
    // 0) keep finite
    std::vector<double> x; x.reserve(x_raw.size());
    for (double v : x_raw) if (std::isfinite(v)) x.push_back(v);
    if (x.size() < 8) {
        if (x.empty()) return -std::numeric_limits<double>::infinity();
        return (*std::min_element(x.begin(), x.end())) - 1e-12; // no deletion
    }

    // 1) Silverman bandwidth with IQR guard
    auto silverman_bandwidth = [&](const std::vector<double>& v)->double {
        const size_t n = v.size();
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / double(n);
        double var = 0.0; for (double a : v) var += (a - mean) * (a - mean);
        const double sd = std::sqrt(var / double(n - 1));
        std::vector<double> tmp = v;
        std::nth_element(tmp.begin(), tmp.begin() + n/4, tmp.end());
        const double q1 = tmp[n/4];
        std::nth_element(tmp.begin(), tmp.begin() + (3*n)/4, tmp.end());
        const double q3 = tmp[(3*n)/4];
        const double iqr = q3 - q1;
        const double s = (sd > 0.0 && iqr > 0.0) ? std::min(sd, iqr/1.34) : std::max(sd, iqr/1.34);
        double h = 0.9 * s * std::pow(double(n), -1.0/5.0);
        if (!(h > 1e-9)) h = 1e-9;
        return h;
    };
    const double h = silverman_bandwidth(x);

    // 2) Build grid adaptively and compute Gaussian KDE (normalized to integrate ~1)
    const double xmin = *std::min_element(x.begin(), x.end());
    const double xmax = *std::max_element(x.begin(), x.end());
    const double pad = 3.0; // domain padding in multiples of bandwidth
    const double lo = xmin - pad * h, hi = xmax + pad * h;

    // Adaptive grid size: ~points_per_bw per bandwidth across the effective span
    const double effective_span = (hi - lo);
    const double points_per_bw = 48.0;           // resolution per bandwidth (tune 32~64 if needed)
    int grid_size = int(std::ceil((effective_span / h) * points_per_bw));
    if (grid_size < 512)  grid_size = 512;
    if (grid_size > 8192) grid_size = 8192;

    std::vector<double> grid(grid_size);
    const double dx = (hi - lo) / double(grid_size - 1);
    for (int i = 0; i < grid_size; ++i) grid[i] = lo + dx * double(i);

    std::vector<double> f(grid_size, 0.0);
    const double inv_h = 1.0 / h;
    const double inv_sqrt_2pi = 0.3989422804014327; // 1/sqrt(2*pi)
    for (int j = 0; j < grid_size; ++j) {
        double sum = 0.0, gj = grid[j];
        for (double xi : x) {
            const double u = (gj - xi) * inv_h;
            sum += std::exp(-0.5 * u * u) * inv_sqrt_2pi;
        }
        f[j] = (sum / double(x.size())) * inv_h;
    }
    // normalize by trapezoid rule
    double area = 0.0;
    for (int j = 1; j < grid_size; ++j) area += 0.5 * (f[j] + f[j-1]) * (grid[j] - grid[j-1]);
    if (area > 0.0) for (double& v : f) v /= area;

    // 3) global mode index
    int i_mode = int(std::max_element(f.begin(), f.end()) - f.begin());
    const double f_mode = f[i_mode];

    // 4) binary search threshold c so that the connected component around the mode has mass ≈ p
    auto mass_of_modal_component = [&](double c, int& l_idx, int& r_idx)->double {
        int L = i_mode, R = i_mode;
        while (L - 1 >= 0           && f[L-1] >= c) --L;
        while (R + 1 < grid_size    && f[R+1] >= c) ++R;
        l_idx = L; r_idx = R;
        double m = 0.0;
        for (int j = L + 1; j <= R; ++j) m += 0.5 * (f[j] + f[j-1]) * (grid[j] - grid[j-1]);
        return m;
    };
    double lo_c = 0.0, hi_c = f_mode, best_c = lo_c;
    int best_l = 0, best_r = grid_size - 1;
    for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo_c + hi_c);
        int l_idx = 0, r_idx = 0;
        const double mass = mass_of_modal_component(mid, l_idx, r_idx);
        if (mass >= p) { // region too wide -> increase c to shrink
            lo_c = mid; best_c = mid; best_l = l_idx; best_r = r_idx;
        } else {
            hi_c = mid;
        }
    }
    const int l_idx = best_l;
    const double L = grid[l_idx];

    // 5) protection guards: (a) small density ratio at L; (b) clear valley between L and mode
    const double f_L = f[l_idx];
    const double ratio = (f_mode > 0.0 ? f_L / f_mode : 1.0);

    double valley = f_L;
    for (int j = l_idx; j <= i_mode; ++j) valley = std::min(valley, f[j]);
    const double valley_ratio = (f_mode > 0.0 ? valley / f_mode : 1.0);

    if (ratio > ratio_thresh || valley_ratio > valley_thresh) {
        // Not truly sparse or no separation -> no deletion
        return xmin - 1e-12;
    }
    // Accept deleting left tail strictly below L
    return L;
}

}
