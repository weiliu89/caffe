import caffe
import numpy

a = numpy.load('/volNAS/share/models/scene/34_scene_v1/mean.npy')

print 'Array dim:', a.shape
print 'array[0:2][0][0] = (%f, %f, %f)' % (a[0][0][0], a[1][0][0], a[2][0][0])

a2 = numpy.array([a])

b = caffe.io.array_to_blobproto(a2)

print 'Blob dim:', b.shape.dim

f = open('/home/yoco/mean.binaryproto', 'wb')
f.write(b.SerializeToString())
f.close()
