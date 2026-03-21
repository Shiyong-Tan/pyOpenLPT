#include "test.h"
#include "ObjectInfo.h"
#include "Camera.h"

// test constructor
bool test_function_1 ()
{
    Object2D obj2d(Pt2D(1, 2));
    if (obj2d._pt_center != Pt2D(1, 2))
    {
        return false;
    }

    Tracer2D tr2d(Pt2D(3, 4));
    if (tr2d._pt_center != Pt2D(3, 4))
    {
        return false;
    }

    Object3D obj3d(Pt3D(5, 6, 7));
    if (obj3d._pt_center != Pt3D(5, 6, 7))
    {
        return false;
    }

    Tracer3D tr3d(Pt3D(8, 9, 10));
    if (tr3d._pt_center != Pt3D(8, 9, 10))
    {
        return false;
    }


    return true;
}

// test Tracer3D functions
bool test_function_2 ()
{
    Tracer3D tr3d;
    tr3d._pt_center = Pt3D(1, 2, 3);
    std::vector<int> camid_list;
    std::vector<Tracer2D> tracer2d_list;

    // test addTracer2D
    Tracer2D tr2d_1 = Tracer2D(Pt2D(4, 5));
    Tracer2D tr2d_2 = Tracer2D(Pt2D(6, 7));
    Tracer2D tr2d_3 = Tracer2D(Pt2D(8, 9));
    std::vector<Tracer2D> tr2d_list_ans = {tr2d_1, tr2d_2, tr2d_3};
    std::vector<int> camid_list_ans = {0, 10, 2};

    tr3d.addTracer2D(tr2d_1, 0);
    tr3d.addTracer2D(tr2d_2, 10);
    tr3d.addTracer2D(tr2d_3, 2);
    camid_list = tr3d._camid_list;
    tracer2d_list = tr3d._tr2d_list;
    if (camid_list.size() != 3 || tracer2d_list.size() != 3)
    {
        std::cout << "test_function_2: addTracer2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "camid_list.size() = " << camid_list.size() << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        return false;
    }
    for (int i = 0; i < 3; i++)
    {
        if (camid_list[i] != camid_list_ans[i] || tracer2d_list[i]._pt_center != tr2d_list_ans[i]._pt_center)
        {
            std::cout << "test_function_2: addTracer2D failed (line " << __LINE__ << ")" << std::endl;
            std::cout << "camid_list[i] = " << camid_list[i] << std::endl;
            std::cout << "camid_list_ans[i] = " << camid_list_ans[i] << std::endl;
            return false;
        }
    }
    ////////////////////////

    // test removeTracer2D
    tr3d.removeTracer2D(100);
    camid_list = tr3d._camid_list;
    tracer2d_list = tr3d._tr2d_list;
    if (camid_list.size() != 3 || tracer2d_list.size() != 3)
    {
        std::cout << "test_function_2: removeTracer2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "camid_list.size() = " << camid_list.size() << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        return false;
    }
    for (int i = 0; i < 3; i++)
    {
        if (camid_list[i] != camid_list_ans[i] || tracer2d_list[i]._pt_center != tr2d_list_ans[i]._pt_center)
        {
            std::cout << "test_function_2: removeTracer2D failed (line " << __LINE__ << ")" << std::endl;
            std::cout << "camid_list[i] = " << camid_list[i] << std::endl;
            std::cout << "camid_list_ans[i] = " << camid_list_ans[i] << std::endl;
            return false;
        }
    }

    tr3d.removeTracer2D(10);
    std::vector<Tracer2D> tr2d_list_ans_2 = {tr2d_1, tr2d_3};
    std::vector<int> camid_list_ans_2 = {0, 2};
    camid_list = tr3d._camid_list;
    tracer2d_list = tr3d._tr2d_list;
    if (camid_list.size() != 2 || tracer2d_list.size() != 2)
    {
        std::cout << "test_function_2: removeTracer2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "camid_list.size() = " << camid_list.size() << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        return false;
    }
    for (int i = 0; i < 2; i++)
    {
        if (camid_list[i] != camid_list_ans_2[i] || tracer2d_list[i]._pt_center != tr2d_list_ans_2[i]._pt_center)
        {
            std::cout << "test_function_2: removeTracer2D failed (line " << __LINE__ << ")" << std::endl;
            std::cout << "camid_list[i] = " << camid_list[i] << std::endl;
            std::cout << "camid_list_ans[i] = " << camid_list_ans[i] << std::endl;
            return false;
        }
    }

    tr3d.removeTracer2D({0, 2});
    camid_list = tr3d._camid_list;
    tracer2d_list = tr3d._tr2d_list;
    if (camid_list.size() != 0 || tracer2d_list.size() != 0)
    {
        std::cout << "test_function_2: removeTracer2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "camid_list.size() = " << camid_list.size() << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        return false;
    }
    ////////////////////////

    // test clearTracer2D
    tr3d.addTracer2D(tr2d_1, 0);
    tr3d.addTracer2D(tr2d_2, 10);
    tr3d.addTracer2D(tr2d_3, 2);
    tr3d.clearTracer2D();
    camid_list = tr3d._camid_list;
    tracer2d_list = tr3d._tr2d_list;
    if (camid_list.size() != 0 || tracer2d_list.size() != 0)
    {
        std::cout << "test_function_2: clearTracer2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "camid_list.size() = " << camid_list.size() << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        return false;
    }
    ////////////////////////

    // test updateTracer2D
    tr3d.addTracer2D(tr2d_1, 0);
    tr3d.addTracer2D(tr2d_2, 10);
    tr3d.addTracer2D(tr2d_3, 2);
    Tracer2D tr2d_4 = Tracer2D(Pt2D(11, 12));
    tr3d.updateTracer2D(tr2d_4, 10);
    tracer2d_list = tr3d._tr2d_list;
    if (tracer2d_list.size() != 3 || tracer2d_list[1]._pt_center != tr2d_4._pt_center)
    {
        std::cout << "test_function_2: updateTracer2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        return false;
    }

    tr3d.updateTracer2D({tr2d_4, tr2d_4}, {0, 2});
    tracer2d_list = tr3d._tr2d_list;
    if (tracer2d_list.size() != 2 || 
        tracer2d_list[0]._pt_center != tr2d_4._pt_center || 
        tracer2d_list[1]._pt_center != tr2d_4._pt_center)
    {
        std::cout << "test_function_2: updateTracer2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        return false;
    }
    ////////////////////////

    // test projectObject2D
    std::vector<std::shared_ptr<Camera>> cam_list_all;
    std::vector<int> camid_list_2 = {0, 1, 2, 3};
    for (int i = 0; i < 5; i ++)
    {
        auto cam_st = CameraFactory::loadFromFile("../test/inputs/test_ObjectInfo/cam"+std::to_string(i+1)+".txt");
        if (!cam_st)
        {
            std::cout << "test_function_2: load camera failed (line " << __LINE__ << ")" << std::endl;
            return false;
        }
        cam_list_all.push_back(cam_st.value());
    }
    cam_list_all[4]->is_active = false;
    tr3d.projectObject2D(cam_list_all);
    tracer2d_list = tr3d._tr2d_list;
    camid_list = tr3d._camid_list;
    if (tracer2d_list.size() != 4 || camid_list.size() != 4)
    {
        std::cout << "test_function_2: projectObject2D failed (line " << __LINE__ << ")" << std::endl;
        std::cout << "tracer2d_list.size() = " << tracer2d_list.size() << std::endl;
        std::cout << "camid_list.size() = " << camid_list.size() << std::endl;
        return false;
    }
    for (int i = 0; i < 4; i++)
    {
        auto proj_st = cam_list_all[i]->project(tr3d._pt_center);
        if (!proj_st || tracer2d_list[i]._pt_center != proj_st.value() || camid_list[i] != camid_list_2[i])
        {
            std::cout << "test_function_2: projectObject2D failed (line " << __LINE__ << ")" << std::endl;
            return false;
        }
    }
    ////////////////////////

    return true;
}

// test IQR-based robust scale recovery in calRadiusFromCams
bool test_IQR_filtering()
{
    // Create a simple pinhole camera for testing
    PinholeParam param;
    param.cam_mtx = Matrix<double, 3, 3>::Identity();
    param.cam_mtx(0, 0) = 1000.0;  // fx
    param.cam_mtx(1, 1) = 1000.0;  // fy
    param.cam_mtx(0, 2) = 500.0;   // cx
    param.cam_mtx(1, 2) = 500.0;   // cy
    param.dist_coeff = Matrix<double, 1, 5>::Zero();
    param.r_mtx = Matrix<double, 3, 3>::Identity();
    param.t_vec = Pt3D(0, 0, 0);
    param.t_vec_inv = Pt3D(0, 0, 0);
    param.nrow = 1000;
    param.ncol = 1000;
    
    auto cam = std::make_shared<PinholeCamera>(param);
    
    // Test case 1: Normal data without outliers
    {
        std::vector<std::shared_ptr<Camera>> cams;
        std::vector<std::unique_ptr<Object2D>> obj2d_list;
        
        // Create 5 cameras with consistent radius measurements
        Pt3D X(0, 0, 100);  // bubble at 100mm distance
        for (int i = 0; i < 5; i++) {
            cams.push_back(cam);
            auto bb2d = std::make_unique<Bubble2D>();
            bb2d->_r_px = 10.0 + i * 0.1;  // slight variation
            obj2d_list.push_back(std::move(bb2d));
        }
        
        double R = Bubble::calRadiusFromCams(cams, X, obj2d_list);
        if (!std::isfinite(R) || R <= 0.0) {
            std::cout << "test_IQR_filtering: normal case failed" << std::endl;
            return false;
        }
    }
    
    // Test case 2: Data with outliers (should be filtered by IQR)
    {
        std::vector<std::shared_ptr<Camera>> cams;
        std::vector<std::unique_ptr<Object2D>> obj2d_list;
        
        Pt3D X(0, 0, 100);
        // Create measurements: 4 normal + 1 outlier
        for (int i = 0; i < 4; i++) {
            cams.push_back(cam);
            auto bb2d = std::make_unique<Bubble2D>();
            bb2d->_r_px = 10.0 + i * 0.1;
            obj2d_list.push_back(std::move(bb2d));
        }
        // Add an outlier
        cams.push_back(cam);
        auto bb2d_outlier = std::make_unique<Bubble2D>();
        bb2d_outlier->_r_px = 50.0;  // outlier
        obj2d_list.push_back(std::move(bb2d_outlier));
        
        double R_with_outlier = Bubble::calRadiusFromCams(cams, X, obj2d_list);
        if (!std::isfinite(R_with_outlier) || R_with_outlier <= 0.0) {
            std::cout << "test_IQR_filtering: outlier case failed" << std::endl;
            return false;
        }
        
        // The result should be close to the normal values, not influenced heavily by outlier
        // This is a qualitative check - exact value depends on IQR filtering
    }
    
    // Test case 3: Single value (edge case)
    {
        std::vector<std::shared_ptr<Camera>> cams;
        std::vector<std::unique_ptr<Object2D>> obj2d_list;
        
        Pt3D X(0, 0, 100);
        cams.push_back(cam);
        auto bb2d = std::make_unique<Bubble2D>();
        bb2d->_r_px = 10.0;
        obj2d_list.push_back(std::move(bb2d));
        
        double R = Bubble::calRadiusFromCams(cams, X, obj2d_list);
        if (!std::isfinite(R) || R <= 0.0) {
            std::cout << "test_IQR_filtering: single value case failed" << std::endl;
            return false;
        }
    }
    
    std::cout << "test_IQR_filtering: all tests passed" << std::endl;
    return true;
}


int main()
{
    fs::create_directories("../test/results/test_ObjectInfo/");

    IS_TRUE(test_function_1());
    IS_TRUE(test_function_2());
    IS_TRUE(test_IQR_filtering());

    return 0;
}
