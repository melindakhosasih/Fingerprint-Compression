import cv2
import numpy as np
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import math

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type

class FingerprintFeatureExtractor(object):
    def __init__(self):
        self._mask = []
        self._skel = []
        self.minutiaeTerm = []
        self.minutiaeBif = []
        self.minutiaeDot = []
        self._spuriousMinutiaeThresh = 10

    def setSpuriousMinutiaeThresh(self, spuriousMinutiaeThresh):
        self._spuriousMinutiaeThresh = spuriousMinutiaeThresh

    def __skeletonize(self, img):
        img = np.uint8(img > 128)
        self._skel = skimage.morphology.skeletonize(img)
        self._skel = np.uint8(self._skel) * 255
        self._mask = img * 255

    def __computeAngle(self, block, minutiaeType):
        angle = []
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        if (minutiaeType.lower() == 'termination'):
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
                        if (sumVal > 1):
                            angle.append(float('nan'))
            return (angle)

        elif (minutiaeType.lower() == 'bifurcation'):
            (blkRows, blkCols) = np.shape(block)
            CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
            angle = []
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
            if (sumVal != 3):
                angle.append(float('nan'))
            return (angle)
        elif minutiaeType.lower() == 'dot':
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if block[i][j] != 0:
                        angle.append(float('nan'))  # Dot minutiae have no direction, so assign NaN angle
                        sumVal += 1
            if sumVal != 1:
                angle.append(float('nan'))
            return angle

    def __getTerminationBifurcationDotIsland(self):
        self._skel = self._skel == 255
        (rows, cols) = self._skel.shape
        self.minutiaeTerm = np.zeros(self._skel.shape)
        self.minutiaeBif = np.zeros(self._skel.shape)
        self.minutiaeDot = np.zeros(self._skel.shape)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (self._skel[i][j] == 1):
                    block = self._skel[i - 1:i + 2, j - 1:j + 2]
                    block_val = np.sum(block)
                    if (block_val == 2):
                        self.minutiaeTerm[i, j] = 1
                    elif (block_val == 4):
                        self.minutiaeBif[i, j] = 1
                    elif (block_val == 1 and i >= 8 and j >= 8):
                        bigger_block = self._skel[i - 8: i + 7, j - 8:j + 7]
                        if (np.sum(bigger_block) > 5):
                            self.minutiaeDot[i, j] = 1

        self._mask = convex_hull_image(self._mask > 0)
        self._mask = erosion(self._mask, square(5))  # Structuing element for mask erosion = square(5)
        self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm


    def __removeSpuriousMinutiae(self, minutiaeList, img):
        img = img * 0
        SpuriousMin = []
        numPoints = len(minutiaeList)
        D = np.zeros((numPoints, numPoints))
        for i in range(1,numPoints):
            for j in range(0, i):
                (X1,Y1) = minutiaeList[i]['centroid']
                (X2,Y2) = minutiaeList[j]['centroid']

                dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
                D[i][j] = dist
                if(dist < self._spuriousMinutiaeThresh):
                    SpuriousMin.append(i)
                    SpuriousMin.append(j)

        SpuriousMin = np.unique(SpuriousMin)
        for i in range(0,numPoints):
            if(not i in SpuriousMin):
                (X,Y) = np.int16(minutiaeList[i]['centroid'])
                img[X,Y] = 1

        img = np.uint8(img)
        return(img)

    def __cleanMinutiae(self, img):
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(self.minutiaeTerm)
        self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img))

        # self.minutiaeDot = skimage.measure.label(self.minutiaeDot, connectivity=2)
        # RP = skimage.measure.regionprops(self.minutiaeDot)
        # self.minutiaeDot = self.__removeSpuriousMinutiae(RP, np.uint8(img))

    def __performFeatureExtraction(self):
        FeaturesTerm = []
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

        WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
        FeaturesTerm = []
        for num, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Termination')
            if(len(angle) == 1):
                FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

        FeaturesBif = []
        self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
        WindowSize = 1  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Bifurcation')
            if(len(angle) == 3):
                FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))

        FeaturesDot = []
        self.minutiaeDot = skimage.measure.label(self.minutiaeDot, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeDot))
        WindowSize = 1  # For Dot, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Dot')
            # if len(angle) == 1:
            FeaturesDot.append(MinutiaeFeature(row, col, angle, 'Dot'))


        return (FeaturesTerm, FeaturesBif, FeaturesDot)

    def extractMinutiaeFeatures(self, img):
        self.__skeletonize(img)

        self.__getTerminationBifurcationDotIsland()

        self.__cleanMinutiae(img)

        FeaturesTerm, FeaturesBif, FeaturesDot = self.__performFeatureExtraction()
        return(FeaturesTerm, FeaturesBif, FeaturesDot)

    def showResults(self, FeaturesTerm, FeaturesBif, FeaturesDot):
        
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255*self._skel
        DispImg[:, :, 1] = 255*self._skel
        DispImg[:, :, 2] = 255*self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

            # # Draw the line indicating the orientation
            # orientation_angle = np.radians(curr_minutiae.Orientation)
            # orientation_length = 10  # Length of the orientation line
            # end_row = int(row - orientation_length * np.sin(orientation_angle))
            # end_col = int(col + orientation_length * np.cos(orientation_angle))
            # cv2.line(DispImg, (col, row), (end_col, end_row), (0, 255, 0), 2)

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

            # # Draw the lines indicating the orientations
            # orientation_angles = curr_minutiae.Orientation
            # orientation_length = 10  # Length of the orientation lines
            # for angle in orientation_angles:
            #     end_row = int(row - orientation_length * np.sin(angle))
            #     end_col = int(col + orientation_length * np.cos(angle))
            #     cv2.line(DispImg, (col, row), (end_col, end_row), (0, 255, 0), 1)

        for idx, curr_minutiae in enumerate(FeaturesDot):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 255, 0))
        
        cv2.imshow('output', DispImg)
        cv2.waitKey(0)

    def saveResult(self, img_name, FeaturesTerm, FeaturesBif, FeaturesDot):
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255 * self._skel
        DispImg[:, :, 1] = 255 * self._skel
        DispImg[:, :, 2] = 255 * self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

            # # Draw the line indicating the orientation
            # orientation_angle = np.radians(curr_minutiae.Orientation)
            # orientation_length = 10  # Length of the orientation line
            # end_row = int(row - orientation_length * np.sin(orientation_angle))
            # end_col = int(col + orientation_length * np.cos(orientation_angle))
            # cv2.line(DispImg, (col, row), (end_col, end_row), (0, 255, 0), 2)

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

            # # Draw the lines indicating the orientations
            # orientation_angles = curr_minutiae.Orientation
            # orientation_length = 10  # Length of the orientation lines
            # for angle in orientation_angles:
            #     end_row = int(row - orientation_length * np.sin(angle))
            #     end_col = int(col + orientation_length * np.cos(angle))
            #     cv2.line(DispImg, (col, row), (end_col, end_row), (0, 255, 0), 1)

        for idx, curr_minutiae in enumerate(FeaturesDot):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 255, 0))

        cv2.imwrite('./minutiae/' + img_name, DispImg)


def extract_minutiae_features(img, img_name, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False):
    feature_extractor = FingerprintFeatureExtractor()
    feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if (invertImage):
        img = 255 - img

    FeaturesTerm, FeaturesBif, FeaturesDot = feature_extractor.extractMinutiaeFeatures(img)

    if (saveResult):
        feature_extractor.saveResult(img_name, FeaturesTerm, FeaturesBif, FeaturesDot)

    if(showResult):
        feature_extractor.showResults(FeaturesTerm, FeaturesBif, FeaturesDot)

    return(FeaturesTerm, FeaturesBif, FeaturesDot)



def calculate_score(features_1, features_2):
    if len(features_1) == 0 and len(features_2) == 0:
        return 0, 0
    
    total_match = 0
    matches = set()

    for minutiae_2 in features_2:
        key = (minutiae_2.locX, minutiae_2.locY)
        matches.add(key)

    for minutiae_1 in features_1:
        key = (minutiae_1.locX, minutiae_1.locY)
        if key in matches:
            total_match += 1
        else:
            # Check for minutiae points within 1 pixel of "key"
            x, y = key
            neighbors = [(x + dx, y + dy) for dx in range(-1, 2) for dy in range(-1, 2)]
            if any(neighbor in matches for neighbor in neighbors):
                total_match += 1

    return total_match, len(features_1)
