defmodule VGG16Model do
  @moduledoc """
  VGG16Model
  """

  @shape_size 224
  @shape_depth 1

  @reshape_size {@shape_depth, @shape_size,  @shape_size}

  require Axon
  require Nx

  alias Evision, as: OpenCV

  defguardp is_list_gt_zero(x) when is_list(x) and length(x) > 0
  defguardp is_input_shape(x) when is_tuple(x) and tuple_size(x) == 4
  defguard is_trainable(x, y, z) when is_function(x) and is_integer(y) and is_integer(z)

  @spec block_1() :: %Axon{}
  defp block_1() do
    Axon.input({nil, 1, 224, 224}, "input")
    |> Axon.conv(64, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv1_1")
    |> Axon.conv(64, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv1_2")
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2], name: "max_pool_1")
  end

  @spec block_2(%Axon{}) :: %Axon{}
  defp block_2(%Axon{} = block) do
    block
    |> Axon.conv(128, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv2_1")
    |> Axon.conv(128, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv2_2")
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2], name: "max_pool_2")
  end

  @spec block_3(%Axon{}) :: %Axon{}
  defp block_3(%Axon{} = block) do
    block
    |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_1")
    |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_2")
    |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_3")
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2], name: "max_pool_3")
  end

  @spec block_4(%Axon{}) :: %Axon{}
  defp block_4(%Axon{} = block) do
    block
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_1")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_2")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_3")
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2], name: "max_pool_4")
  end

  @spec block_5(%Axon{}) :: %Axon{}
  defp block_5(%Axon{} = block) do
    block
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_1")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_2")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_3")
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2], name: "max_pool_5")
  end

  @spec block_encoder(%Axon{}, Integer) :: %Axon{}
  defp block_encoder(%Axon{} = block, units) when is_integer(units) and units > 0 do
    block
    |> Axon.flatten(name: "flatten")
    |> Axon.dense(4096, activation: :relu, name: "fc_1")
    |> Axon.dropout(rate: 0.5, name: "dropout_1")
    |> Axon.dense(4096, activation: :relu, name: "fc_2")
    |> Axon.dropout(rate: 0.5, name: "dropout_2")
    |> Axon.dense(units, activation: :softmax, name: "output")
  end

  @doc """
  Build VGG16 model.

  ## Examples
      model = Vgg16Model.build_model(1011)
  """
  @spec build_model(Integer) :: %Axon{}
  def build_model(units) when is_integer(units) do
    block_1()
    |> block_2
    |> block_3
    |> block_4
    |> block_5
    |> block_encoder(units)
  end

  @spec augment_tensor(Nx.Tensor) :: Nx.Tensor
  defp augment_tensor(img_tensor) do
    t =
      img_tensor
      |> Nx.reshape(@reshape_size)
      |> Nx.divide(255.0)

    if Nx.random_uniform({}) > 0.5 do
      Nx.reverse(t, axes: [0])
    else
      Nx.reverse(t, axes: [1])
    end
  end

  @spec image_handler(String.t()) :: Nx.Tensor
  defp image_handler(filename) do
    OpenCV.imread(filename, flags: OpenCV.cv_IMREAD_GRAYSCALE())
    |> then(fn {:ok, val} -> val end)
    |> OpenCV.resize([224, 224])
    |> then(fn {:ok, val} -> val end)
    |> OpenCV.Nx.to_nx()
    |> augment_tensor()
  end

  @spec parse_image(String.t()) :: Nx.Tensor
  defp parse_image(filename) when is_bitstring(filename) do
    # read in image at filename, resize image to 224x224, convert to
    # Nx.tensor, reshape tensor to @reshape_size, and then convert
    # pixels to 255.0
    filename
    |> image_handler()
    |> augment_tensor()
  end

  @spec model_serialize(%Axon{}, Map) :: {:ok, Binary} | {:error, String.t()}
  defp model_serialize(%Axon{} = model, state) when is_map(state) do
    # serialize the model for writing it to file
    try do
      content = Axon.serialize(model, state)
      {:ok, content}
    catch
      error, reason -> {:error, "Caught #{reason}: #{error}"}
    end
  end

  @doc """
  Convert a list of file name strings to a tensor

  ## Parameters
    - images: List of file names of JPG images

  ## Examples
      tensor = Vgg16Model.process_images(images)
  """
  @spec process_images(list(String.t())) :: Nx.Tensor
  def process_images(images) when is_list_gt_zero(images) do
    # take in a list file names and pass them to parse_image before putting
    # them in a stack (Nx.stack).
    images
    |> Enum.map(fn image -> parse_image(image) end)
    |> Nx.stack
  end

  @spec parse_label(Integer, Integer) :: Nx.Tensor
  defp parse_label(label, size) when is_integer(label) and is_integer(size) do
    # convert label (Integer) into a Nx.tensor
    Nx.equal(label, Nx.tensor(Enum.to_list(1..size)))
  end

  @doc """
  Take in a list of labels and parse them. Then add them to stack (Nx.stack)

  ## Parameters
    - labels: List of labels (Integers)
    - size: Size of tensor for labels

  ## Examples
    tensor = process_labels([1, 2, 3], 3)
  """
  @spec process_labels(list(Integer), Integer) :: Nx.Tensor
  def process_labels(labels, size) when is_list_gt_zero(labels) and is_integer(size) do
    labels
    |> Enum.map(fn label -> parse_label(label, size) end)
    |> Nx.stack
  end

  @doc """
  Train VGG16 model.

  ## Examples
      data = data("./data/location")
      model = VGG16Model.build_model(1011)
      epochs = 10

      state = VGG16Model.train(model, data, epochs)
  """
  @spec train_model(%Axon{}, Function, Integer, Integer) :: Map
  def train_model(%Axon{} = model, data, epochs, iterations)
  when is_trainable(data, epochs, iterations) do
    optimizer = Axon.Optimizers.adam(0.001)

    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, optimizer, log: 1)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: iterations, timeout: :infinity)
  end

  @doc """
  Test VGG16 model.

  ## Example
    data = data("./data/location")
    model = VGG16Model.build_model(1011)
    state = VGG16Model.train(model, data, 1011)

    Vgg16Model.test(model, state, data)
  """
  @spec test_model(%Axon{}, Map, Function) :: nil
  def test_model(%Axon{} = model, state, data) when is_map(state) and is_function(data) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(data, state)
  end

  @doc """
  Save model to file.

  ## Parameters
    - filepath: String representing the location to save model
    - model: Term representing the Axon model that was built by the user
    - state: Term representing the Axon parameters after model has been trained

  ## Example
    case VGG16Model.save_model(filepath, model, state) do
      {:ok} -> IO.write("File saved successfully\n")
      {:error, reason} -> IO.write(reason)
    end
  """
  @spec save_model(String.t(), %Axon{}, Map) :: :ok | {:error, String.t()}
  def save_model(filepath, %Axon{} = model, state) when is_bitstring(filepath) and is_map(state) do
    case model_serialize(model, state) do
      {:ok, content} -> File.write(filepath, content)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Load model from file.

  ## Parameters
    - filepath: String that represents the file to load

  ## Example
      case VGG16Model.load_model(filepath) do
        {:ok, model} -> handle_model(model)
        {:error, reason} -> IO.write(reason)
      end
  """
  @spec load_model(String.t()) :: {:ok, {%Axon{}, Map}} | {:error, String.t()}
  def load_model(filepath) when is_bitstring(filepath) do
    case File.read(filepath) do
      {:ok, binary} -> Axon.deserialize(binary)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Make prediction given an image.

    ## Parameters
      - model: the vgg16 model
      - model_state: the trained model parameters
      - paths: a list of paths to JPG images

    ## Example
      data = data("./data/location")
      model = VGG16Model.build_model({nil, 224, 224, 3}, 10)
      state = VGG16Model.train(model, data, 10)

      image_path = "./local/dir/filename"
      pred = VGG16Model.predict(model, state, image_path)
      IO.inspect pred
  """
  @spec predict(%Axon{}, Map, list(String.t())) :: Nx.Tensor
  def predict(%Axon{} = model, model_state, paths)
  when is_map(model_state) and is_list_gt_zero(paths) do
    images = process_images(paths)
    Axon.predict(model, model_state, images)
  end
end

VGG16Model.build_model(1011) |> IO.inspect
