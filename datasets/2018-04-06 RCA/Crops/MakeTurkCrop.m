%% Load the image

im_cy3 = imread('mip_cy3.tif');
im_cy5 = imread('mip_cy5.tif');

%% Create the image (cy3)
x = 925;
y = 649;

roi_widths = [500];

for i = 1:length(roi_widths)

    im_cropped = imcrop(im_cy3, [x y (roi_widths(i) - 1) (roi_widths(i) - 1)]);
    
    filename = sprintf('cy3_%dpxroi.tif', roi_widths(i));

    imwrite(im_cropped, filename)

end

%% Create the image (cy5)
x = 1;
y = 1213;

roi_widths = [100, 150, 200, 250, 300];

for i = 1:length(roi_widths)

    im_cropped = imcrop(im_cy5, [x y (roi_widths(i) - 1) (roi_widths(i) - 1)]);
    
    filename = sprintf('cy5_%dpxroi.tif', roi_widths(i));

    imwrite(im_cropped, filename)

end

%%


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