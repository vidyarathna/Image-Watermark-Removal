import cv2
import numpy as np

def remove_watermark(image, watermark):
    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints_image, descriptors_image = orb.detectAndCompute(gray_image, None)
    keypoints_watermark, descriptors_watermark = orb.detectAndCompute(gray_watermark, None)

    # Match descriptors
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_image, descriptors_watermark, None)

    # Convert matches tuple to list
    matches = list(matches)

    # Sort matches by distance
    matches.sort(key=lambda x: x.distance)

    # Remove top matches (adjust this threshold according to your requirement)
    good_matches = matches[:int(len(matches) * 0.15)]

    # Extract keypoints
    image_pts = np.float32([keypoints_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    watermark_pts = np.float32([keypoints_watermark[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, _ = cv2.findHomography(watermark_pts, image_pts, cv2.RANSAC)

    # Warp watermark image
    watermark_warped = cv2.warpPerspective(watermark, H, (image.shape[1], image.shape[0]))

    # Inpaint the watermark area
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(image_pts), (255, 255, 255))
    mask = cv2.dilate(mask, None, iterations=10)
    removed_watermark = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return removed_watermark

# Load images
image = cv2.imread('input_images\\water.jpg')
watermark = cv2.imread('input_images\\Rattan Garden Lounge.png')

# Remove watermark
removed_watermark = remove_watermark(image, watermark)

# Resize images for display
resized_image = cv2.resize(image, (800, 600))  # Adjust the dimensions as needed
resized_removed_watermark = cv2.resize(removed_watermark, (800, 600))  # Adjust the dimensions as needed

# Display results
cv2.imshow('Original Image', resized_image)
cv2.imshow('Removed Watermark', resized_removed_watermark)
cv2.waitKey(0)
cv2.destroyAllWindows()

