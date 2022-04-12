from setuptools import setup
import os
from glob import glob
package_name = 'ceiling_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name+"/UNET/VGG16", package_name+"/utils"],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('config/*.yaml')),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hossein',
    maintainer_email='hsa150@sfu.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ceiling_segmentation_start = ceiling_segmentation.main:main',
        ],
    },
)
