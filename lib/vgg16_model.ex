defmodule VGG16Model do
  @moduledoc """
  VGG16Model
  """

  require Axon
  require Nx
  require StbImage

  defp block_1(input_shape) do
    input_shape
    |>Axon.input("input")
    |> Axon.conv(64, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv1_1")
    |> Axon.conv(64, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv1_2")
    |> Axon.max_pool(strides: [2, 2], name: "max_pool_1")
  end

  defp block_2(block) do
    block
    |> Axon.conv(128, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv2_1")
    |> Axon.conv(128, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv2_2")
    |> Axon.max_pool(strides: [2, 2], name: "max_pool_2")
  end

  defp block_3(block) do
    block
    |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_1")
    |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_2")
    |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_3")
    |> Axon.max_pool(strides: [2, 2], name: "max_pool_3")
  end

  defp block_4(block) do
    block
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_1")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_2")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_3")
    |> Axon.max_pool(strides: [2, 2], name: "max_pool_4")
  end

  defp block_5(block) do
    block
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_1")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_2")
    |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_3")
    |> Axon.max_pool(strides: [2, 2], name: "max_pool_4")
  end

  defp block_encoder(block, count) do
    block
    |> Axon.flatten(name: "flatten")
    |> Axon.dense(4096, activation: :relu, name: "fc_1")
    |> Axon.dropout(rate: 0.5, name: "dropout_1")
    |> Axon.dense(4096, activation: :relu, name: "fc_2")
    |> Axon.dropout(rate: 0.5, name: "dropout_4")
    |> Axon.dense(count, activation: :softmax, name: "output")
  end

  defp model_serialize(model, state) do
    try do
      content = Axon.serialize(model, state)
      {:ok, content}
    catch
      error, reason -> {:error, "Caught #{reason}: #{error}"}
    end
  end

  defp parse_image(filename) do
    filename
    |> StbImage.read_file!
    |> StbImage.resize(224, 224)
    |> StbImage.to_nx
    |> Nx.reshape({224, 224, 3})
    |> Nx.divide(255.0)
  end

  @doc """
  Convert a list of file name strings to a tensor

  ## Parameters
    - images: List of file names of JPG images

  ## Examples
      tensor = Vgg16Model.process_images(images)
  """
  @spec process_images(list(String.t())) :: Nx.Tensor
  def process_images(images) do
    images
    |> Enum.map(fn image -> parse_image(image) end)
    |> Nx.stack
  end

  defp label_list_size(binary_list) do
    size = length(binary_list)
    if size < 10 do
      s = 10 - size
      binary_list ++ List.duplicate(0, s)
    else
      binary_list
    end
  end

  defp parse_label(label) do
    label
    |> String.to_integer
    |> Integer.digits(2)
    |> label_list_size
    |> Nx.tensor(type: {:u, 8})
  end

  @spec process_labels(list(String.t())) :: Nx.Tensor
  def process_labels(labels) do
    labels
    |> Enum.map(fn label -> parse_label(label) end)
    |> Nx.stack
  end

  @doc """
  Build VGG16 model.

  ## Examples
      output_count = 10
      input_shape = {nil, 224, 224, 3}

      model = Vgg16Model.build_model(input_shape, output_count)
  """
  @spec build_model(tuple(), integer()) :: term()
  def build_model(input_shape, count) do
    input_shape
    |> block_1
    |> block_2
    |> block_3
    |> block_4
    |> block_5
    |> block_encoder(count)
  end

  @doc """
  Train VGG16 model.

  ## Examples
      data = data("./data/location")
      model = VGG16Model.build_model({nil, 224, 224, 3}, 10)
      epochs = 10

      state = VGG16Model.train(model, data, epochs)
  """
  @spec train_model(term(), term(), integer()) :: term()
  def train_model(model, data, epochs) do
    optimizer = Axon.Optimizers.adam(1.0e-4)

    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, optimizer, log: 1)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: 1)
  end

  @doc """
    Test VGG16 model.

    ## Example
      data = data("./data/location")
      model = VGG16Model.build_model({nil, 224, 224, 3}, 10)
      state = VGG16Model.train(model, data, 10)

      Vgg16Model.test(model, state, data)
   """
   @spec test_model(term(), term(), term()) :: nil
   def test_model(model, state, data) do
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
   @spec save_model(String.t(), term(), term()) :: :ok | {:error, String.t()}
   def save_model(filepath, model, state) do
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
   @spec load_model(String.t()) :: term() | {:error, String.t()}
   def load_model(filepath) do
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
   @spec predict(term(), term(), list(String.t())) :: Nx.Tensor
   def predict(model, model_state, paths) do
     images = process_images(paths)
     Axon.predict(model, model_state, images)
   end
end

VGG16Model.build_model({nil, 224, 224, 3}, 10)
