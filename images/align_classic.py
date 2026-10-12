import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images(template_path, scene_path, use_sift=True, min_match_count=10):
    # Load in grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    scene = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    if template is None or scene is None:
        raise FileNotFoundError("Could not load one or both images.")

    # Feature detection
    if use_sift:
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        norm_type = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(template, None)
    kp2, des2 = detector.detectAndCompute(scene, None)
    print(f"Template keypoints: {len(kp1)}, Scene keypoints: {len(kp2)}")

    if des1 is None or des2 is None:
        raise ValueError("No descriptors found. Images may lack texture.")

    # Matching
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(f"Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < min_match_count:
        raise ValueError(f"Not enough matches: {len(good_matches)}/{min_match_count}")

    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography estimation failed.")
    print(f"RANSAC inliers: {int(mask.sum())}/{len(good_matches)}")

    # Warp template onto scene
    h, w = template.shape
    warped = cv2.warpPerspective(template, H, (scene.shape[1], scene.shape[0]))
    return warped, H, mask, template, scene, kp1, kp2, good_matches

def visualize(template, scene, warped, kp1, kp2, matches):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(template, cmap='gray'); axes[0].set_title("Template (I/O Plate)"); axes[0].axis('off')
    axes[1].imshow(scene, cmap='gray'); axes[1].set_title("Scene (Socket)"); axes[1].axis('off')
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(warped, 50, 150)
    scene_color[edges > 0] = [0, 0, 255]
    axes[2].imshow(cv2.cvtColor(scene_color, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Warped Template Overlaid"); axes[2].axis('off')
    plt.tight_layout(); plt.show()

    match_img = cv2.drawMatches(template, kp1, scene, kp2, matches, None,
                                matchColor=(0,255,0), singlePointColor=None, flags=2)
    plt.figure(figsize=(15,6)); plt.imshow(match_img)
    plt.title("Feature Matches"); plt.axis('off'); plt.show()

if __name__ == "__main__":
    warped, H, mask, template, scene, kp1, kp2, matches = align_images(
        "io_plate.jpg", "socket_scene.jpg", use_sift=True
    )
    print("\nHomography Matrix H:\n", H)
    visualize(template, scene, warped, kp1, kp2, matches)
    cv2.imwrite("aligned_plate.png", warped)
    print("Saved aligned_plate.png")