"""Test protobuf compilation and functionality."""


def test_protobuf_import() -> None:
    """Test that protobuf files can be imported."""
    import proto.data_loader_config_pb2 as data_loader_config_pb2
    import proto.model_config_pb2 as model_config_pb2
    import proto.root_config_pb2 as root_config_pb2
    import proto.training_config_pb2 as training_config_pb2
    import proto.training_metrics_pb2 as training_metrics_pb2

    # Test creating config objects
    data_loader_config = data_loader_config_pb2.DataLoaderConfig()
    assert data_loader_config is not None

    root_config = root_config_pb2.RootConfig()
    assert root_config is not None

    training_config = training_config_pb2.TrainingConfig()
    assert training_config is not None

    model_config = model_config_pb2.ModelConfig()
    assert model_config is not None

    metrics = training_metrics_pb2.DataLoaderMetricsProto()
    assert metrics is not None


def test_protobuf_functionality() -> None:
    """Test basic protobuf functionality."""
    import proto.data_loader_config_pb2 as data_loader_config_pb2
    import proto.root_config_pb2 as root_config_pb2

    # Create a config and set some values
    config = data_loader_config_pb2.DataLoaderConfig()
    config.file_path_provider.directory = "/test/path"

    # Serialize and deserialize
    serialized = config.SerializeToString()
    assert len(serialized) > 0

    config2 = data_loader_config_pb2.DataLoaderConfig()
    config2.ParseFromString(serialized)

    assert config2.file_path_provider.directory == "/test/path"

    # Test RootConfig functionality
    root_config = root_config_pb2.RootConfig()
    root_config.name = "test_config"
    root_config.data_loader.file_path_provider.directory = "/test/path"

    # Serialize and deserialize root config
    root_serialized = root_config.SerializeToString()
    assert len(root_serialized) > 0

    root_config2 = root_config_pb2.RootConfig()
    root_config2.ParseFromString(root_serialized)

    assert root_config2.name == "test_config"
    assert root_config2.data_loader.file_path_provider.directory == "/test/path"
