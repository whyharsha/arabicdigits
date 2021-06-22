using DataFrames
using CSV
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA

function getimages(filename)
    filepath = pwd() * "/images/" * filename

    mtrx = Matrix(DataFrame(CSV.File(filepath)))

    return mtrx'
end

function getlabels(filename)
    filepath = pwd() * "/images/" * filename
    vec(Matrix(DataFrame(CSV.File(filepath))))
end

function load_data(args)
    train_data_file = "csvTrainImages.csv"
    test_data_file = "csvTestImages.csv"
    train_label_file = "csvTrainLabel.csv"
    test_label_file = "csvTestLabel.csv"

    train_data = getimages(train_data_file)
    test_data = getimages(test_data_file)
    train_labels = getlabels(train_label_file)
    test_labels = getlabels(test_label_file)

    xtrain = Flux.flatten(train_data)
    xtest = Flux.flatten(test_data)

    ytrain, ytest = onehotbatch(train_labels, 1:28), onehotbatch(test_labels, 1:28)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

function build_model(; imgsize=(32,32,1), nclasses=28)
    return Chain(
 	        Dense(prod(imgsize), 32, relu),
            Dense(32, nclasses))
end

function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += logitcrossentropy(model(x), y, agg=sum)
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))
        num +=  size(x, 2)
    end
    return ls / num, acc / num
end

@kwdef mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 256    # batch size
    epochs::Int = 10        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

function train(; kws...)
    args = Args(; kws...) # collect options in a struct for convenience

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Create test and train dataloaders
    train_loader, test_loader = load_data(args)

    # Construct model
    model = build_model() |> device
    ps = Flux.params(model) # model's trainable parameters
    
    ## Optimizer
    opt = ADAM(args.η)

    train_loss = 0.0f0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    
    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x), device(y) # transfer data to device
            gs = gradient(() -> logitcrossentropy(model(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end
        
        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
    end

    return train_loss, train_acc, test_loss, test_acc
end

#run training
train()
