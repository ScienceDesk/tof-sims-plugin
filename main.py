##  Required imports
import matplotlib; matplotlib.use('Agg')
from sdesk.proc import io
from matplotlib import pyplot, gridspec
from tofsims_segmentation import get_segmented_img
import pySPM


def main():
    # LOAD INPUT FILE
    input_metadata = io.get_input_metadata()
    files = io.get_input_files(input_metadata)
    sdesk_input_file = files[0]
    file_metadata = input_metadata[0]

    # PROCESS THE INPUT FILE AND PRODUCE RESULTS
    spectrum = pySPM.ITA(sdesk_input_file.path())
    TOF_collection = pySPM.ITA_collection(sdesk_input_file.path())
    TOF_collection.run_pca()
    TOF_pca = TOF_collection.PCA.get_pca_col(3, True)
    output_np_list = [ get_segmented_img(TOF_pca[ch]) for (i, ch) in enumerate(TOF_pca.channels)] 
    
    # CREATE THE OUTPUT FILES 
    sdesk_output_file = io.create_output_file('full_spectrum.png')
    fig, ax = pyplot.subplots()
    spectrum.show_spectrum(ax=ax)
    fig.savefig(sdesk_output_file.path())
     
    PC1_np = output_np_list[1]
    sdesk_output_file = io.create_output_file('PC1.png')
    pyplot.imsave(sdesk_output_file.path(), PC1_np, cmap='bwr')

    PC2_np = output_np_list[3]
    sdesk_output_file = io.create_output_file('PC2.png')
    pyplot.imsave(sdesk_output_file.path(), PC2_np, cmap='bwr')
    
    # THUMBNAIL IMAGE
    thumbnail_fig = pyplot.figure(figsize=(12, 12), tight_layout=True)
    gs = gridspec.GridSpec(2, 2)
    ax0 = thumbnail_fig.add_subplot(gs[0, :])
    ax1 = thumbnail_fig.add_subplot(gs[1, 0])   
    ax2 = thumbnail_fig.add_subplot(gs[1, 1]) 

    sdesk_output_file = io.create_output_file('_thumbnail_picture.png')
    spectrum.show_spectrum(ax=ax0)
    ax1.imshow(PC1_np, cmap='bwr')
    ax2.imshow(PC2_np, cmap='bwr')
    thumbnail_fig.savefig(sdesk_output_file.path())
    

# Call method main()
main()