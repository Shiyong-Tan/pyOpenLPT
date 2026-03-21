
#include "Track.h"

Track::Track(std::unique_ptr<Object3D> obj3d, int t)
{ 
    _obj3d_list.emplace_back(std::move(obj3d)); //unique_ptr can only be moved
    _t_list.push_back(t); 
}


void Track::addNext(std::unique_ptr<Object3D> obj3d, int t)
{
    _obj3d_list.emplace_back(std::move(obj3d));
    _t_list.push_back(t);
}

void Track::saveTrack(std::ostream& output, int track_id)
{    
    for (size_t i = 0; i < _obj3d_list.size(); ++i)
    {
        output << track_id << "," << _t_list[i] << ","; // FrameID is integer

        // Delegate the rest to the 3D object
        _obj3d_list[i]->saveObject3D(output);
    }
}

void Track::loadTrack(std::ifstream& fin, const ObjectConfig& cfg,
                      const std::vector<std::shared_ptr<Camera>>& camera_models)
{
    std::string line;
    int my_tid = -1;

    while (true) {
        std::streampos pos = fin.tellg();
        if (!std::getline(fin, line)) break;
        if (line.empty()) continue;

        std::istringstream row(line);

        std::string s_tid, s_fid;
        if (!std::getline(row, s_tid, ',') || !std::getline(row, s_fid, ',')) continue;

        int tid = -1, fid = 0;
        try { tid = std::stoi(s_tid); fid = std::stoi(s_fid); }
        catch (...) { continue; }

        if (my_tid == -1) {
            my_tid = tid;
        } else if (tid != my_tid) {
            fin.clear();
            fin.seekg(pos);
            break;
        }

        CreateArgs a;
        auto obj = cfg.creatObject3D(std::move(a));
        obj->loadObject3D(row);
        obj->_is_tracked = true;
        obj->projectObject2D(camera_models);

        _obj3d_list.emplace_back(std::move(obj));
        _t_list.emplace_back(fid);
    }
}
