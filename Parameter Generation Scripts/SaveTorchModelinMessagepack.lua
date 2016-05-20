require 'torch'
require 'nn'
mp = require 'MessagePack'	
mp.set_array'without_hole'


--**********************************Initialize Path *********************************--
--************************************Parameters*************************************--

--Path to load torch nn model
load_path = "/home/user/Desktop/testModel.nn"

--Root path of saving directory
--NOTE: THE GIVEN DIRECTORY SHOULD EXIST, otherwise you will face 'bad argument' error 
save_path = "/home/user/Desktop/parameters/"

--***********************************************************************************--
--***********************************************************************************--

print("Loading nn model: " .. load_path)

net = torch.load(load_path)

print("nn model: ")
print(net)

--save weight and bias of convolution layers
conv_nodes = net:findModules('nn.SpatialConvolution')
for i = 1 , #conv_nodes do
	tensor_weight = conv_nodes[i].weight:type('torch.FloatTensor')
	tensor_bias = conv_nodes[i].bias:type('torch.FloatTensor')
	
	weight = {}
	for j = 1 , tensor_weight:size(1) do
		weight[j] = {}
		for k = 1 , tensor_weight:size(2) do
			weight[j][k] = {}
			for l = 1 , tensor_weight:size(3) do
				weight[j][k][l] = {}
				for m = 1 , tensor_weight:size(4) do
					weight[j][k][l][m] = tensor_weight[j][k][l][m]
				end
			end		
  		end
	end
	
	bias = {}
	for i = 1 , tensor_bias:size(1) do
		bias[i] = tensor_bias[i]
	end

	mp_w = mp.pack(weight)
	mp_b = mp.pack(bias)

	file = io.open(save_path.."model_param_conv"..i..".msg" , "w")
	file:write(mp_w)
	file:write(mp_b)
	file:close()
	
	print("saved convolution layer " .. i .. " parameters: " .. "'model_param_conv" .. i .. ".msg'")
end

--save weight and bias of fully-connected layers
conv_nodes = net:findModules('nn.Linear')
for i = 1 , #conv_nodes do
	tensor_weight = conv_nodes[i].weight:type('torch.FloatTensor')
	tensor_bias = conv_nodes[i].bias:type('torch.FloatTensor')
	
	weight = {}
	l = 1
	for j = 1 , tensor_weight:size(1) do
		for k = 1 , tensor_weight:size(2) do
			weight[l] = tensor_weight[j][k]
			l = l + 1
  		end
	end

	
	bias = {}
	for i = 1 , tensor_bias:size(1) do
		bias[i] = tensor_bias[i]
	end

	mp_w = mp.pack(weight)
	mp_b = mp.pack(bias)

	file = io.open(save_path.."model_param_fc"..i..".msg" , "w")
	file:write(mp_w)
	file:write(mp_b)
	file:close()
	
	print("saved fully-connected layer " .. i .. " parameters: " .. "'model_param_fc" .. i .. ".msg'")
end


