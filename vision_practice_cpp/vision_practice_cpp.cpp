#include <iostream>
#include "geomaching.h"
#include "GeoMatch.h"

#define PYRAMID_DOWN_LEVEL 1

String path_template = "imgSrc/template.bmp";
String path_template_flip = "imgSrc/template-flip.bmp";
String path_source = "imgSrc/sample-2.bmp";
geomaching GeoMatching;

int main()
{
	GeoMatching.addgeomachingModel(path_template, "Side A");
	GeoMatching.addgeomachingModel(path_template_flip, "Side B");
	GeoMatching.sourceStream(path_source);
	double time = GeoMatching.Matching();
	//GeoMatching.addGeoMatchModel(path_template, "Side A");
	//GeoMatching.addGeoMatchModel(path_template_flip, "Side B");
	//GeoMatching.setImageSource(path_source);
	//GeoMatching.matching(true);
	waitKey(0);
}

