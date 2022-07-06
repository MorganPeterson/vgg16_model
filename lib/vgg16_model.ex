defmodule VGG16Model do
  @moduledoc """
  Documentation for `Vgg16Model`.
  """

  require Axon

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

  defp process_image(image_path) do
    image_path
    |> StbImage.read_file!
    |> StbImage.resize(224, 224)
    |> StbImage.to_nx
    |> Nx.reshape({1, 224, 224, 3})
    |> Nx.divide(255.0)
  end

  @doc """
  Build VGG16 model.

  ## Examples
      output_count = 10
      input_shape = {nil, 224, 224, 3}

      model = Vgg16Model.build_model(input_shape, output_count)
  """
  def build_model!(input_shape, count) do
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
  def train_model!(model, data, epochs) do
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
  def test_model!(model, state, data) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(data, state)
  end

  @doc """
  Save model to file.

  ## Example
      data = data("./data/location")
      model = VGG16Model.build_model({nil, 224, 224, 3}, 10)
      state = VGG16Model.train(model, data, 10)

      filepath = "./local/dir/filename"
      VGG16Model.save_model(filepath, model, state)
  """
  def save_model(filepath, model, state) do
    case model_serialize(model, state) do
      {:ok, content} -> File.write(filepath, content)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Load model from file.

  ## Example
      filepath = "./local/dir/filename"
      VGG16Model.save_model(filepath)
  """
  def load_model(filepath) do
    case File.read(filepath) do
      {:ok, binary} -> Axon.deserialize(binary)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Make prediction given an image.

  ## Example
      data = data("./data/location")
      model = VGG16Model.build_model({nil, 224, 224, 3}, 10)
      state = VGG16Model.train(model, data, 10)

      image_path = "./local/dir/filename"
      pred = VGG16Model.predict(model, state, image_path)
      IO.inspect pred
  """
  def predict(model, model_state, image_path) do
    image = process_image(image_path)
    Axon.predict(model, model_state, image)
  end
end

VGG16Model.build_model!({nil, 224, 224, 3}, 1100)
