import caffe
import numpy

a = numpy.load('/volNAS/share/models/scene/34_scene_v1/mean.npy')

print 'Array dim:', a.shape
a2 = numpy.swapaxes(a, 0, 1)
print 'Array dim after swap:', a2.shape

b = caffe.io.array_to_blobproto(a2)

print 'Blob dim:', b.shape.dim

f = open('/home/yoco/mean.binaryproto', 'wb')
f.write(b.SerializeToString())
f.close()
