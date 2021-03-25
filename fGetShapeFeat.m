function shapeFeatVector = fGetShapeFeat(region)
% This function gets the shapeFeat descriptors from a binary region
%
% INPUT
%   - region: Image with a binary region to get the descriptors from. There
%   is ONLY ONE REGION IN THE IMAGE. The type of the image must be LOGICAL
%
% OUTPUT
%   shapeFeatVector: Vector with all the descriptors of the ShapeFeat.
%   These are:
%     · Convex Area: Number of pixels of the smallest convex polygon that
%     contains the region.
%     · Eccentricity: Scalar that specifies the eccentricity of the ellipse 
%     that has the same second central moments as the region. 
%     · Perimeter: Number of points in the contour of the region.
%     · Equivalent Diameter: Scalar that specifies the diameter of a circle
%      with the same area as the region.
%     · Extent: Scalar that specifies the ratio of the pixels of the
%     region to the pixels in the bounding box around the region.
%     · Filled Area: Number of pixels belonging to the region after filling
%     its possible holes.
%     · Minor Axis Length: Length of the minor axis of the ellipse that has
%     the same second central moments as the region. 
%     · Major Axis Length: Length of the major axis of the ellipse that has
%     the same second central moments as the region.
%     · R: Ratio between the major and minor axis of the ellipse that has
%     the same second central moments as the region.
%     · Solidity: Ratio between the area of the region and the Convex Area
%     of the region.
%

% Initialise the output
shapeFeatVector = zeros(1,10);

% ====================== YOUR CODE HERE ======================
    stats = regionprops(region,'Area','Eccentricity','Perimeter','EquivDiameter','Extent',...
       'FilledArea','MinorAxisLength','MajorAxisLength','Solidity');
% ============================================================

% Convex Area: Number of pixels of the smallest convex polygon that
% contains the region.
% ============================================================
shapeFeatVector(1) = stats.Area;
% ============================================================

% Eccentricity: Scalar that specifies the eccentricity of the ellipse that
% has the same second central moments as the region.
% ============================================================
shapeFeatVector(2) = stats.Eccentricity;
% ============================================================

% Perimeter: Number of points in the contour of the region.
% ============================================================
shapeFeatVector(3) = stats.Perimeter;
% ============================================================

% Equivalent Diameter: Scalar that specifies the diameter of a circle with
% the same area as the region.
% ============================================================
shapeFeatVector(4) = stats.EquivDiameter;
% ============================================================

% Extent: Scalar that specifies the ratio of the pixels of the region to
% the pixels in the bounding box around the region.
% ============================================================
shapeFeatVector(5) = stats.Extent;
% ============================================================

% Filled Area: Number of pixels belonging to the region after filling its
% possible holes.
% ============================================================
shapeFeatVector(6) = stats.FilledArea;
% ============================================================

% Minor Axis Length: Length of the minor axis of the ellipse that has the
% same second central moments as the region.
% ============================================================
shapeFeatVector(7) = stats.MinorAxisLength;
% ============================================================

% Major Axis Length: Length of the major axis of the ellipse that has the
% same second central moments as the region.
% ============================================================
shapeFeatVector(8) = stats.MajorAxisLength;
% ============================================================

% R: Ratio between the major and minor axis of the ellipse that has the
% same second central moments as the region.
% ============================================================
shapeFeatVector(9) = shapeFeatVector(7)/shapeFeatVector(8);
% ============================================================

% Solidity: Ratio between the area of the region and the Convex Area of the
% region.
% ============================================================
shapeFeatVector(10) = stats.Solidity;
% ============================================================


end