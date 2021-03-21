# -*- coding: utf-8 -*-

import pathlib
import json
import time
import os
import sys
import tempfile
import dateutil.parser
from collections import OrderedDict

import arcpy


LANDSAT8_BANDS = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]
COEFF_WETNESS = dict(
    zip(LANDSAT8_BANDS, [1, 0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559])
)


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Timber Harvets"
        self.alias = "timberharvests"

        # List of tool classes associated with this toolbox
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Estimate clear cuts from Landsat 8 imagery"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        source_dir = arcpy.Parameter(
            name="source_dir",
            displayName="Directory with folders of Landsat 8 L2 captures",
            direction="Input",
            datatype="DEFolder",
            parameterType="Required",
        )
        extent_area = arcpy.Parameter(
            name="extent_layer",
            displayName="Extent to clip imagery to",
            direction="Input",
            datatype="GPFeatureLayer",
            parameterType="Required",
        )
        inbetweens = arcpy.Parameter(
            name="add_inbetweens",
            displayName="Add layers of processing steps to map",
            direction="Input",
            datatype="GPBoolean",
            parameterType="Required",
        )

        params = [source_dir, extent_area, inbetweens]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """."""
        params = {p.name: p.valueAsText for p in parameters}
        messages.addMessage("Here we go")
        params["add_inbetweens"] = True if "true" in params["add_inbetweens"] else False

        self.main(**params)

    def main(self, *args, **params):
        tmpdir = tempfile.TemporaryDirectory()
        arcpy.env.scratchWorkspace = tmpdir.name

        arcpy.SetProgressor("default", "Looking for Landsat metadata...")

        out_dir = pathlib.Path(params["source_dir"], "results")
        out_dir.mkdir(exist_ok=True)
        self.out_dir = str(out_dir.absolute())
        self.add_inbetweens = params["add_inbetweens"]
        arcpy.env.workspace = self.out_dir
        arcpy.env.overwrite = True
        arcpy.env.overwriteOutput = True
        arcpy.env.pyramid = "None"

        self.map = None
        try:
            p = arcpy.mp.ArcGISProject("CURRENT")
        except OSError:
            # Arcgis not running
            pass
        else:
            available_maps = p.listMaps()
            if available_maps:
                self.map = available_maps[-1]

        metadata = self.find_landsat_captures(params["source_dir"])

        layers = []
        for i, (source_dir, desc) in enumerate(metadata):
            # Set the workspace to the Landsat subdirectory to make it easy to reference each image by name
            arcpy.SetProgressor(
                "step",
                "Compositing and processing Landsat images",
                0,
                len(metadata),
                i + 1,
            )
            images = self.load_landsat_capture(source_dir, desc)
            composite = self.process_landsate_capture(
                images,
                prefix=pathlib.Path(source_dir).name,
                output_bands=["red", "green", "blue", "ndvi", "wetness"],
                extent_layer=params["extent_layer"],
            )
            if self.add_inbetweens and self.map:
                # add full composite with RGB to map so we can see it
                self.map.addDataFromPath(composite)

            layers.append((composite, desc))

        if not layers:
            arcpy.AddError("No Landsat images found")
            return None

        # reduce to the bands of interest for faster processing
        diffs = self.make_diffs(layers, band_of_interest="wetness")
        features = self.classify(diffs)
        self.composite_features(features)

    def find_landsat_captures(self, source_dir):
        metadata = []
        for d in pathlib.Path(source_dir).glob("LC08_L2SP_*_02_T1"):
            for f in d.glob("*_02_T1_SR_stac.json"):
                desc = json.load(f.open())
                arcpy.AddMessage(f"Found {f.name} with {len(desc.keys())} keys")
                metadata.append((d.absolute(), desc))
        # Sort by date
        metadata.sort(key=lambda md: md[1]["properties"]["datetime"])
        return metadata

    def load_landsat_capture(self, source_dir, desc):
        assets = desc["assets"]
        band_data = desc["properties"]["eo:bands"]
        images_by_band = {}
        arcpy.AddMessage(f"Searching {source_dir.name}")
        for image_path in pathlib.Path(source_dir).glob("*_02_T1_SR_*.TIF"):
            band_suffix = image_path.name.split("_02_T1_")[-1]
            if band_suffix in assets:
                name = assets[band_suffix]["title"]
                if "eo:bands" in assets[band_suffix]:
                    bands = [band_data[i] for i in assets[band_suffix]["eo:bands"]]
                    # arcpy.AddMessage(f"Found {name}")
                    for band in bands:
                        images_by_band[band["common_name"]] = arcpy.Raster(
                            str(image_path.absolute())
                        )

        arcpy.AddMessage(f"Total bands found: {len(list(images_by_band))}")

        return images_by_band

    def rescale(self, raster):
        # Can only work on one band at a time
        return (raster - raster.minimum) / (raster.maximum - raster.minimum)

    def zscores(self, raster):
        # Can only work on one band at a time
        return arcpy.Raster((raster - raster.mean) / raster.standardDeviation)

    def process_landsate_capture(
        self, images, prefix, output_bands, extent_layer=None, skip_existing=True,
    ):

        # @TODO this should use a hash of the band names or something unique
        num_bands = len(output_bands)

        result_name = os.path.join(
            self.out_dir, f"{prefix}_AOI_{num_bands}BAND_COMPOSITE.TIF"
        )

        if skip_existing and pathlib.Path(result_name).exists():
            arcpy.AddMessage(
                f"Using existing {num_bands} band composite: {result_name}"
            )
            return arcpy.Raster(result_name)

        images = self.add_derived_bands(images)

        # Filter bands to those selected
        # Keep order that was specificed in output_bands
        images_sorted = OrderedDict()
        for band in output_bands:
            if band in images:
                images_sorted[band] = images[band]
        images = images_sorted
        arcpy.AddMessage(images.keys())

        bands = list(images.keys())
        rasters = list(images.values())

        arcpy.AddMessage(f"Compositing {len(rasters)} bands into one raster")
        result = arcpy.management.CompositeBands(rasters, None)

        arcpy.AddMessage(f"Clipping raster to extent")
        result = arcpy.management.Clip(
            result[0],
            "",
            None,
            extent_layer,
            "3.4e+38",
            "ClippingGeometry",
            "NO_MAINTAIN_EXTENT",
        )

        img = arcpy.Raster(result[0])
        arcpy.AddMessage(f"Renaming bands")
        for i, name in enumerate(img.bandNames):
            # This doesn't work when using a geodatabase?!
            img.renameBand(i + 1, bands[i])

        arcpy.AddMessage(f"Saved initial band composite as {result_name}")

        arcpy.AddMessage(f"Colormap info:  {img.getColormap()}")
        arcpy.AddMessage(f"Variables:  {img.variables}")
        arcpy.AddMessage(f"Bands in result:  {img.bandNames}")

        img.save(result_name)

        return img

    def add_derived_bands(self, image):
        arcpy.AddMessage("Adding derived bands")

        image["ndvi"] = (image["nir08"] - image["red"]) / (
            image["nir08"] + image["red"]
        )

        image["wetness"] = sum(
            [
                image["blue"] * COEFF_WETNESS["blue"],
                image["green"] * COEFF_WETNESS["green"],
                image["red"] * COEFF_WETNESS["red"],
                image["nir08"] * COEFF_WETNESS["nir08"],
                image["swir16"] * COEFF_WETNESS["swir16"],
                image["swir22"] * COEFF_WETNESS["swir22"],
            ]
        )

        return image

    def make_diffs(self, layers, band_of_interest):
        arcpy.SetProgressor(
            "step", "Calculating diffs between images", 0, len(layers), 1
        )
        diffs = []
        for i, (image_one, desc) in enumerate(layers):
            if i < len(layers) - 1:
                image_two, next_desc = layers[i + 1]
                single_band_one = self.zscores(
                    image_one.getRasterBands(band_of_interest)
                )
                single_band_two = self.zscores(
                    image_two.getRasterBands(band_of_interest)
                )
                diff = self.zscores(single_band_one - single_band_two)
                name = (
                    desc["properties"]["datetime"]
                    + "_"
                    + next_desc["properties"]["datetime"]
                    + "_zscore_diff_zscore.tif"
                )
                out_path = os.path.join(self.out_dir, name)
                diff.save(out_path)
                diffs.append((name, (desc, next_desc)))
                if self.add_inbetweens and self.map:
                    self.map.addDataFromPath(out_path)
        return diffs

    def classify_with_ecd(self, diff):
        # No longer used 

        image_path, (desc_one, desc_two) = diff
        classifier_ecd = self.params["ecd_classifier"]
        classified = arcpy.ia.ClassifyRaster(image_path, classifier_ecd, None,)

        # Class "3" are USUALLY the clear cuts in this ECD
        cuts_only = arcpy.ddd.Reclassify(
            classified, "Classvalue", "0 NODATA;1 NODATA;2 NODATA;3 1", None, "NODATA"
        )

        cuts_only = arcpy.sa.MajorityFilter(cuts_only, "EIGHT", "HALF")
        cuts_only = arcpy.sa.MajorityFilter(cuts_only, "EIGHT", "HALF")
        cuts_only = arcpy.sa.BoundaryClean(cuts_only)

        return cuts_only

    def classify_with_segment_mean_shift(self, diff, band_of_interest):
        # No longer used

        image_path, (desc_one, desc_two) = diff
        image = arcpy.Raster(image_path)

        # Segment mean shift can handle multiple bands too
        band_num = image.bandNames.index(band_of_interest) + 1

        arcpy.AddMessage(f"Selecting band, num: {band_num}")
        arcpy.AddMessage("Running segment mean shift")
        
        image = arcpy.CopyRaster_management(image, None, pixel_type="8_BIT_UNSIGNED")
        segmented = arcpy.ia.SegmentMeanShift(image, 3, 3, 20, band_num, -1);

        if self.inbetweens and self.map:
            self.map.addDataFromPath(segmented)

        cuts_only = arcpy.ddd.Reclassify(segmented, "Value", f"0 {segmented.mean} NODATA;{segmented.mean} 255 1", None, "NODATA")

        if self.inbetweens and self.map:
            self.map.addDataFromPath(cuts_only)

        return cuts_only

    def classify_manually(self, diff, std=2):
        image_path, (desc_one, desc_two) = diff

        date_one = dateutil.parser.parse(desc_one["properties"]["datetime"])
        date_two = dateutil.parser.parse(desc_two["properties"]["datetime"])
        scene_one = desc_one["properties"]["landsat:scene_id"]
        scene_two = desc_two["properties"]["landsat:scene_id"]

        arcpy.AddMessage(f"Classifying {date_one} to {date_two} diff")

        image = arcpy.Raster(image_path)

        # Choose anything over x num std deviations
        zscore_cut_off = str(std)
        cuts_only = arcpy.ddd.Reclassify(
            image,
            "VALUE",
            f"-9999  NODATA;{zscore_cut_off} 9999 {zscore_cut_off}",
            None,
            "NODATA",
        )
        cuts_only = arcpy.sa.BoundaryClean(cuts_only, "NO_SORT", "TWO_WAY")
        if self.add_inbetweens and self.map:
            self.map.addDataFromPath(cuts_only)
        # arr = arcpy.RasterToNumPyArray(cuts_only)
        # Sum pixels, multiple by resolution (30 sq meters), convert to acres
        # acres = arr.sum() * (30*30) * 0.000247105
        # arcpy.AddMessage(f"Total acres cut: {acres}")

        arcpy.AddMessage(f"Raster to polygon")
        shapes = arcpy.conversion.RasterToPolygon(
            cuts_only, None, "NO_SIMPLIFY", "Value", "SINGLE_OUTER_PART", None
        )
        arcpy.AddMessage(f"Adding fields")
        arcpy.management.CalculateField(
            shapes, "date_1", f"'{date_one.date()}'", "PYTHON3", "", "DATE"
        )
        arcpy.management.CalculateField(
            shapes, "date_2", f"'{date_two.date()}'", "PYTHON3", "", "DATE"
        )
        arcpy.management.CalculateField(
            shapes, "name", f"'{date_one.year}-{date_two.year}'", "PYTHON3", "", "TEXT"
        )
        shapes = arcpy.management.Dissolve(
            shapes, None, "name", None, "MULTI_PART", "DISSOLVE_LINES"
        )
        arcpy.management.AddGeometryAttributes(shapes, "AREA", "", "ACRES", None)

        # More fields if we want them
        # arcpy.management.CalculateField(
        #     shapes, "capture_scene_1", f"'{scene_one}'", "PYTHON3", "", "DATE"
        # )
        # arcpy.management.CalculateField(
        #     shapes, "capture_scene_2", f"'{scene_two}'", "PYTHON3", "", "DATE"
        # )

        return shapes

    def classify(self, diffs):

        features = []
        arcpy.SetProgressor("default", "Classifying and converting diffs")
        for diff in diffs:
            features.append(self.classify_manually(diff))
        return features

        # @TODO someday
        # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        #     features = pool.map(classify_one, diffs)

    def composite_features(self, features):

        # Add date & names from individual features.
        fms = arcpy.FieldMappings()
        for feature in features:
            fms.addTable(feature)

        arcpy.AddMessage(str(fms))

        out_name = f"Timber_Harvests_Over_Time_{int(time.time())}"
        out_path = os.path.join(self.out_dir, out_name)
        arcpy.SetProgressor("default", "Creating final shapefile")
        arcpy.AddMessage("Merging features")
        result = arcpy.management.Merge(features, out_name, fms, "ADD_SOURCE_INFO",)
        arcpy.SetProgressor("default", "Adding to map")
        if self.map:
            self.map.addDataFromPath(result[0])

        # @TODO Arc doesn't like this
        # arcpy.FeatureSet(result).save(out_path + ".shp")

    def make_time_series_raster(self, layers):
        # No longer use. Was literally getting "Catastrophic Error" in ArcGIS

        rasters = [image for image, _ in layers]
        names = [image.path for image, _ in layers]
        dates = [desc["properties"]["datetime"] for _, desc in layers]
        arcpy.AddMessage(dates)

        arcpy.AddMessage(f"Making raster collection with {len(rasters)} rasters")

        collection = arcpy.ia.RasterCollection(
            rasters, {"Name": names, "AcquisitionDate": dates}
        )
        mdim_raster = collection.toMultidimensionalRaster(
            variable_field_name="Raster", dimension_field_names="AcquisitionDate"
        )
        arcpy.AddMessage(f"Variables:  {mdim_raster.variables}")
        arcpy.AddMessage(f"Bands in result:  {mdim_raster.bandNames}")
        arcpy.AddMessage(f"Slices:  {mdim_raster.slices}")
        arcpy.AddMessage(f"Workspace:  {arcpy.env.workspace}")

        out_path = str(
            (pathlib.Path(self.out_dir) / "TEST_MULTIDIM_RASTER.crf").absolute()
        )
        arcpy.AddMessage(f"Saving to {out_path}")
        mdim_raster.save(out_path)
        return mdim_raster


if __name__ == "__main__":

    if len(sys.argv) > 1:
        tool = Tool()
        tool.main(
            source_dir=sys.argv[1],
            extent_layer=sys.argv[2],
        )

