#include <iostream>
#include "geomaching.h"

#define PYRAMID_DOWN_LEVEL 1

String path_template = "imgSrc/template.bmp";
String path_template_flip = "imgSrc/template-flip.bmp";

String path_source = "imgSrc/src/Sample-1/sample-1-18.bmp";


geomaching GeoMatching;

int main()
{
	GeoMatching.addgeomachingModel(path_template, "Side A");
	GeoMatching.addgeomachingModel(path_template_flip, "Side B");
	GeoMatching.sourceStream(path_source, PYRAMID_DOWN_LEVEL);
	double time= GeoMatching.Matching();
	cout << "elasped time: " << time << " ms" << endl;
	waitKey(0);
}

