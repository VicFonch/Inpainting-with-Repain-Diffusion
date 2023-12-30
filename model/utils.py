from imports.common_imports import *

def save_figure(sampled_images, original_images, output_path = None, num_images=4, figsize=(10, 7), **kwargs):

    sampled_images = np.array(sampled_images)  
    original_images = np.array(tf.clip_by_value(original_images, 0, 1)) 
    
    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(sampled_images[i], **kwargs)
        plt.axis('off')
        plt.title('Sampleada')
        
        plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(original_images[i], **kwargs)
        plt.axis('off')
        plt.title('Original')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def save_grid(images, output_path):
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    axs = axs.flatten()
    
    images = np.array(images)
    images = (images + 1.)/2.
    
    n_images = images.shape[0]
    for i in range(n_images):
        img = images[i]
        axs[i].imshow(img)
        axs[i].axis("off")
    plt.savefig(output_path)