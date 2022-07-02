EXLA.set_as_nx_default(
  [:tpu, :cuda, :rocm, :host],
  run_options: [keep_on_device: true]
)

defmodule VGG16Model do
  @moduledoc """
  Documentation for `Vgg16Model`.
  """

  require Axon

  @doc """
  Build VGG16 model.

  ## Examples

      iex> Vgg16Model.build_model({224, 224, 3}, 1000)
  """

  def build_model(input, count) do
    block1 =
      Axon.input(input, "input")
      |> Axon.conv(64, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv1_1")
      |> Axon.conv(64, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv1_2")
      |> Axon.max_pool(strides: [2, 2], name: "max_pool_1")

    block2 =
      block1
      |> Axon.conv(128, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv2_1")
      |> Axon.conv(128, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv2_2")
      |> Axon.max_pool(strides: [2, 2], name: "max_pool_2")

    block3 =
      block2
      |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_1")
      |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_2")
      |> Axon.conv(256, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv3_3")
      |> Axon.max_pool(strides: [2, 2], name: "max_pool_3")

    block4 =
      block3
      |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_1")
      |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_2")
      |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv4_3")
      |> Axon.max_pool(strides: [2, 2], name: "max_pool_4")

    block5 =
      block4
      |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_1")
      |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_2")
      |> Axon.conv(512, kernel_size: {3, 3}, padding: :same, activation: :relu, name: "conv5_3")
      |> Axon.max_pool(strides: [2, 2], name: "max_pool_4")

    block5
    |> Axon.flatten(name: "flatten")
    |> Axon.dense(4096, activation: :relu, name: "fc_1")
    |> Axon.dropout(rate: 0.5, name: "dropout_1")
    |> Axon.dense(4096, activation: :relu, name: "fc_2")
    |> Axon.dropout(rate: 0.5, name: "dropout_4")
    |> Axon.dense(count, activation: :softmax, name: "output")
  end

  @doc """
  Train VGG16 model.

  ## Examples

      iex> Vgg16Model.train(model, data, 10)
  """

  def train(model, data, optimizer, epochs) do
    model
      |> Axon.Loop.trainer(:categorical_cross_entropy, optimizer)
      |> Axon.Loop.metric(:accuracy, "Accuracy")
      |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: 100)
  end
end

VGG16Model.build_model({224, 224, 3}, 1100)
