#include <iostream>
#include "matching.h"

#define PYRAMID_DOWN_LEVEL 1

String path_template = "imgSrc/template.bmp";
String path_template_flip = "imgSrc/template-flip.bmp";

String path_source = "imgSrc/src/Sample-1/sample-1-18.bmp";


matching GeoMatching;

int main()
{
	GeoMatching.addmatchingModel(path_template, "Side A");
	GeoMatching.addmatchingModel(path_template_flip, "Side B");
	GeoMatching.sourceStream(path_source, PYRAMID_DOWN_LEVEL);
	GeoMatching.Matching();

	waitKey(0);
}