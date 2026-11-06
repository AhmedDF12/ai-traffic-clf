"""Expected errors a user can fix by changing config, labels, or input files."""


class ProjectError(ValueError):
    """Base class for user-fixable project errors."""


class ConfigError(ProjectError):
    """The YAML configuration is missing, mistyped, or breaks a project rule."""


class LabelMapError(ProjectError):
    """A label-map file is internally inconsistent."""


class UnmappedLabelError(ProjectError):
    """A raw label or capture name has no project class."""


class SchemaError(ProjectError):
    """A table does not match the feature or sidecar contract."""


class SplitError(ProjectError):
    """A train/validation/test assignment is missing or would leak rows."""


class TrackingError(ProjectError):
    """MLflow is unavailable or the local tracking directory cannot be used."""
