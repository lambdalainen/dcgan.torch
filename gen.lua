require 'image'
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

noise = torch.Tensor(64, 100, 1, 1)
--noise = torch.zeros(64, 100, 1, 1)
net = torch.load('bedrooms_4_net_G.t7')

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
    print(net)
end

noise:normal(0, 1)
-- net:float() -- what does this do?

local q = require 'quantize'
net:quantize(q.fixed(1, 8))

-- 64 x 3 x 64 x 64
local images = net:forward(noise)

-- net, net.modules, and net.modules[1] are all tables, but
-- net.modules[1].weight and net.modules[1].output are userdata which cannot be inspected directly

-- images:add(1):mul(0.5) -- what does this do? compare generated images with/without this
image.save('gen.png', image.toDisplayTensor(images))
