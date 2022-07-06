defmodule Vgg16Model.MixProject do
  use Mix.Project

  def project do
    [
      app: :vgg16_model,
      version: "0.1.1",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:axon, "~> 0.2.0-dev", github: "elixir-nx/axon"},
      {:stb_image, "~> 0.5.2"},
      {:exla, "~> 0.2"}
    ]
  end
end
