defmodule Vgg16Model.MixProject do
  use Mix.Project

  def project do
    [
      app: :vgg16_model,
      version: "0.1.3",
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
      {:evision, "~> 0.1.0-dev", github: "cocoa-xu/evision", branch: "main"},
      {:axon, "~> 0.2.0-dev", github: "elixir-nx/axon"},
      {:exla, "~> 0.2"}
    ]
  end
end
