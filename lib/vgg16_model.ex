defmodule VGG16Model do
  @moduledoc """
  Documentation for `Vgg16Model`.
  """

  require Axon
  require EXLA

  @doc """
  Build VGG16 model.

  ## Examples

      iex> Vgg16Model.build_model({224, 224, 3})
  """

  def build_model(input, count) do
    block1 =
      Axon.input(input, "input")
      |> Axon.conv(64, kernel_size: {3}, padding: :same, activation: :relu, name: "conv1_1")
      |> Axon.conv(64, kernel_size: {3}, padding: :same, activation: :relu, name: "conv1_2")
      |> Axon.max_pool(strides: [2], name: "max_pool_1")

    block2 =
      block1
      |> Axon.conv(128, kernel_size: {3}, padding: :same, activation: :relu, name: "conv2_1")
      |> Axon.conv(128, kernel_size: {3}, padding: :same, activation: :relu, name: "conv2_2")
      |> Axon.max_pool(strides: [2], name: "max_pool_2")

    block3 =
      block2
      |> Axon.conv(256, kernel_size: {3}, padding: :same, activation: :relu, name: "conv3_1")
      |> Axon.conv(256, kernel_size: {3}, padding: :same, activation: :relu, name: "conv3_2")
      |> Axon.conv(256, kernel_size: {3}, padding: :same, activation: :relu, name: "conv3_3")
      |> Axon.max_pool(strides: [2], name: "max_pool_3")

    block4 =
      block3
      |> Axon.conv(512, kernel_size: {3}, padding: :same, activation: :relu, name: "conv4_1")
      |> Axon.conv(512, kernel_size: {3}, padding: :same, activation: :relu, name: "conv4_2")
      |> Axon.conv(512, kernel_size: {3}, padding: :same, activation: :relu, name: "conv4_3")
      |> Axon.max_pool(strides: [2], name: "max_pool_4")

    block5 =
      block4
      |> Axon.conv(512, kernel_size: {3}, padding: :same, activation: :relu, name: "conv5_1")
      |> Axon.conv(512, kernel_size: {3}, padding: :same, activation: :relu, name: "conv5_2")
      |> Axon.conv(512, kernel_size: {3}, padding: :same, activation: :relu, name: "conv5_3")
      |> Axon.max_pool(strides: [2], name: "max_pool_4")

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

  def train(model, data, epochs) do
    model_state =
    model
      |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
      |> Axon.Loop.metric(:accuracy, "accuracy")
      |> Axon.Loop.run(data, epochs: epochs, compiler: EXLA)
  end
end

