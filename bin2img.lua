require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local open = io.open

local function read_file(path)
    local file = open(path, "rb") -- r read mode and b binary mode
    if not file then return nil end
    local content = file:read "*a" -- *a or *all reads the whole file
    file:close()
    return content
end

-- http://lua-users.org/lists/lua-l/2010-03/msg00910.html
function convert(x)
  local sign = 1
  local mantissa = string.byte(x, 3) % 128
  for i = 2, 1, -1 do mantissa = mantissa * 256 + string.byte(x, i) end
    if string.byte(x, 4) > 127 then sign = -1 end
    local exponent = (string.byte(x, 4) % 128) * 2 +
                     math.floor(string.byte(x, 3) / 128)
    if exponent == 0 then return 0 end
    mantissa = (math.ldexp(mantissa, -23) + 1) * sign
    return math.ldexp(mantissa, exponent - 127)
end

local output_14 = read_file('bin/output_14_test.bin') -- string
table_14={}
for i = 1, #output_14, 4 do
    local s = output_14:sub(i, i+3)
    table.insert(table_14, convert(s))
end
print(#table_14)

local tensor_14 = torch.reshape(torch.Tensor(table_14), 64, 3, 64, 64)
print(tensor_14:size())

image.save('gen_test.png', image.toDisplayTensor(tensor_14))
