%% Load the image

im = imread('mip_beads.tif');

%% Create the image
x = 1556;
y = 1291;

roi_widths = [100, 150, 200, 250, 300];

for i = 1:length(roi_widths)

    im_cropped = imcrop(im, [x y (roi_widths(i) - 1) (roi_widths(i) - 1)]);
    
    filename = sprintf('beads_%dpxroi.png', roi_widths(i));

    imwrite(im_cropped, filename)

end


%% Load the annotations

annotations = readtable('bead_annotations_20180404.csv');

roi_width = 300;

points = (annotations.row > y) & (annotations.row < (y + roi_width - 1))...
    & (annotations.col > x) & (annotations.col < (x + roi_width - 1));

bounded_annotations = annotations(points, {'row', 'col'});

points = [bounded_annotations.col - x, bounded_annotations.row - y];

%% 
imshow(im_cropped)

hold on
plot(points(:, 1), points(:, 2), 'm*')