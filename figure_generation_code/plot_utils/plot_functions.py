
from matplotlib import patches, image, offsetbox
from PIL import Image
from mpl_toolkits.mplot3d import proj3d
from .plot_defaults import *
import PyPDF2


def get_2d_inh_cmap(n_3d):
    lspc = np.linspace(0, 0.95, n_3d)  # [::-1]
    cmaps = []
    for i in range(n_3d):
        gist_earth = inhibitory_cmap(np.linspace(lspc[i], lspc[i], 10))
        cmaps.append(ListedColormap(gist_earth))
    return cmaps


def set_3d_plot_specs(ax, transparent_panes=True, juggle=True):
    if transparent_panes:  # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.grid(False)
    if juggle:
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    ax.zaxis.set_rotate_label(False)


class Arrow3D(patches.FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        patches.FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def drawCirc(ax, width, height, centX, centY, angle_, theta1_, theta2_, theta_arrow_, color_='black',
             linewidth_=1, headlength_=0.1, headwidth_=0.1, xscale=1):  # xscale is the scale of x relative to y
    # ========Line
    arc = patches.Arc((centX, centY), width, height, angle=angle_, theta1=theta1_, theta2=theta2_,
                      capstyle='butt', linestyle='-', lw=linewidth_, color=color_, clip_on=False)
    ax.add_patch(arc)

    # ========Create the arrow head
    endX = centX + (width / 2) * np.cos(np.radians(theta_arrow_))  # Do trig to determine end position
    endY = centY + (height / 2) * np.sin(np.radians(theta_arrow_))
    # calculate the arrow head as a polygon
    points = [[endX + headwidth_ * (centX - endX), endY + headwidth_ * (centY - endY)],
              [endX - headwidth_ * (centX - endX), endY - headwidth_ * (centY - endY)],
              [endX + headlength_ * xscale * (centY - endY), endY - headlength_ / xscale * (centX - endX)]]
    # clrs = ['red','blue','green','cyan']
    # for i in range(len(points)):
    #    ax.plot([points[i][0]],[points[i][1]],'.',markersize=0.5,c=clrs[i])
    ax.add_patch(patches.Polygon(np.array(points), color=color_, clip_on=False))


# taken from:
# https://stackoverflow.com/questions/44550764/how-can-i-embed-an-image-on-each-of-my-subplots-in-matplotlib
def place_image(im_str, loc="lower left", ax=None, zoom=1, **kw):
    im = image.imread(im_str)
    if ax is None:
        ax = plt.gca()
    imagebox = offsetbox.OffsetImage(im, zoom=zoom*0.72)
    ab = offsetbox.AnchoredOffsetbox(loc=loc, child=imagebox, frameon=False, **kw)
    ax.add_artist(ab)


# taken from:
# https://stackoverflow.com/questions/44550764/how-can-i-embed-an-image-on-each-of-my-subplots-in-matplotlib
def place_image2(im_str, loc="lower left", ax=None, zoom=1, **kw):
    im = Image.open(im_str)
    print(im.size)
    im = im.resize((int(zoom*im.size[0]), int(zoom*im.size[1])), Image.ANTIALIAS)
    if ax is None:
        ax = plt.gca()
    imagebox = offsetbox.OffsetImage(im, zoom=zoom * 0.72)
    ab = offsetbox.AnchoredOffsetbox(loc=loc, child=imagebox, frameon=False, **kw)
    ax.add_artist(ab)


def add_image_to_pdf(pdf_path, image_path, output_path, x, y, width, height):
    # Open the PDF file in read-binary mode
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)

        # Get the first page of the PDF
        page = pdf_reader.getPage(0)

        # Open the image file in read-binary mode
        with open(image_path, 'rb') as image_file:
            image_reader = PyPDF2.PdfFileReader(image_file)
            image_page = image_reader.getPage(0)

            # Scale the image to the specified width and height
            image_page.scaleBy(width / image_page.mediaBox.getWidth(), height / image_page.mediaBox.getHeight())

            # Add the image page to the main PDF page at the specified location (x, y)
            page.mergeTranslatedPage(image_page, x, y)

            # Create a PDF writer object to save the result
            pdf_writer = PyPDF2.PdfFileWriter()
            pdf_writer.addPage(page)

            # Write the modified PDF to the output file
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
