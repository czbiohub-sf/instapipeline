from QuantiusAnnotation import QuantiusAnnotation

json_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/datasets/smFISH_test/smFISH_cells.json'
img_filename = 'C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_300.png'

qa = QuantiusAnnotation(json_filepath, img_filename)

anno_all = qa.df()
qa.print_head(anno_all)