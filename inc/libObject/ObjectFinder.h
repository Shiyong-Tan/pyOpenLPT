#ifndef OBJECTFINDER_H
#define OBJECTFINDER_H

#include <vector>
#include <typeinfo>

#include "Config.h"
#include "ObjectInfo.h"
#include "Matrix.h"
#include "STBCommons.h"
#include "myMATH.h"

#include "CircleIdentifier.h"

class ObjectFinder2D
{
public:
    struct BubbleTileDetection {
        Pt2D center;
        double radius = 0.0;
        double metric = 0.0;
    };

    ObjectFinder2D() = default;
    ~ObjectFinder2D() = default;

    // Find 2D objects in the image based on the object configuration
    std::vector<std::unique_ptr<Object2D>>
    findObject2D(Image const& img, ObjectConfig const& obj_cfg);

    // Bubble-only batch path used by IPR. It preserves the validated tile
    // geometry while scheduling all active camera/tile jobs in one team.
    std::vector<std::vector<std::unique_ptr<Object2D>>>
    findBubble2DFixedBatch(const std::vector<Image>& images,
                           const std::vector<char>& active,
                           const BubbleConfig& cfg);

private:
    std::vector<std::unique_ptr<Object2D>>
    findTracer2D(Image const& img, TracerConfig const& cfg);

    std::vector<std::unique_ptr<Object2D>>
    findBubble2D(Image const& img, BubbleConfig const& cfg);
};


#endif
