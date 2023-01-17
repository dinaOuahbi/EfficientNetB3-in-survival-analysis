/**
 * Script to export a rendered (RGB) image in QuPath v0.2.0.
 *
 * This is much easier if the image is currently open in the viewer,
 * then see https://qupath.readthedocs.io/en/latest/docs/advanced/exporting_images.html
 *
 * The purpose of this script is to support batch processing (Run -> Run for project (without save)),
 * while using the current viewer settings.
 *
 * Note: This was written for v0.2.0 only. The process may change in later versions.
 *
 * @author Pete Bankhead
 */

import qupath.imagej.tools.IJTools
import qupath.lib.gui.images.servers.RenderedImageServer
import qupath.lib.gui.viewer.overlays.HierarchyOverlay
import qupath.lib.regions.RegionRequest

import static qupath.lib.gui.scripting.QPEx.*

// It is important to define the downsample!
// This is required to determine annotation line thicknesses
double downsample = 10

// Request the current viewer for settings, and current image (which may be used in batch processing)
def viewer = getCurrentViewer()
def imageData = getCurrentImageData()

// Add the output file path here
def name =  GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
shortname = name[0..14]
print shortname
String path = buildFilePath('/work/shared/ptbc/CNN_Pancreas_V2/Donnees/EfficientNet_1/', 'rendered', shortname + '.tif')



// Create a rendered server that includes a hierarchy overlay using the current display settings
def server = new RenderedImageServer.Builder(imageData)
    .downsamples(downsample)
    .layers(new HierarchyOverlay(viewer.getImageRegionStore(), viewer.getOverlayOptions(), imageData))
    .build()

// Write or display the rendered image
if (path != null) {
    mkdirs(new File(path).getParent())
    writeImage(server, path)
} else
    IJTools.convertToImagePlus(server, RegionRequest.createInstance(server)).getImage().show()
    
print('Exporting image done  ')