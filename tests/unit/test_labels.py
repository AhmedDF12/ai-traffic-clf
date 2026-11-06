from pathlib import Path

import pytest

from traffic_clf.config import load_config
from traffic_clf.data.labels import (
    SECURITY_CLASSES,
    load_iscx_rules,
    load_security_map,
    map_security_label,
    parse_iscx_capture,
    to_binary_security,
)
from traffic_clf.errors import UnmappedLabelError


@pytest.fixture
def rules(repo_root: Path):
    return load_iscx_rules(load_config(root=repo_root).labels.iscx_vpn)


@pytest.mark.parametrize(
    ("filename", "vpn", "application", "service", "excluded"),
    [
        ("youtube6.pcap", False, "youtube", "streaming", None),
        ("vpn_youtube_A.pcap", True, "youtube", "streaming", None),
        ("netflix1.pcap", False, "netflix", "streaming", None),
        ("spotify2.pcap", False, "spotify", "streaming", None),
        ("vimeo4.pcap", False, "vimeo", "streaming", None),
        ("vpn_skype_files1a.pcap", True, "skype", "file_transfer", None),
        ("skype_file1.pcap", False, "skype", "file_transfer", None),
        ("skype_chat1a.pcap", False, "skype", "chat", None),
        ("skype_audio1.pcap", False, "skype", "voip", None),
        ("hangouts_chat_4a.pcap", False, "hangouts", "chat", None),
        ("hangouts_video2.pcap", False, "hangouts", "voip", None),
        ("vpn_facebook_chat1b.pcap", True, "facebook", "chat", None),
        ("facebook_audio2.pcap", False, "facebook", "voip", None),
        ("facebook_video1.pcap", False, "facebook", "voip", None),
        ("AIMchat2.pcap", False, "aim", "chat", None),
        ("aim_chat_3a.pcap", False, "aim", "chat", None),
        ("icq_chat_3a.pcap", False, "icq", "chat", None),
        ("voipbuster1b.pcap", False, "voipbuster", "voip", None),
        ("ftps_down_1a.pcap", False, "ftps", "file_transfer", None),
        ("sftp_up_2a.pcap", False, "sftp", "file_transfer", None),
        ("scpDown1.pcap", False, "scp", "file_transfer", None),
        ("scp1.pcap", False, "scp", "file_transfer", None),
        ("vpn_bittorrent.pcap", True, "bittorrent", None, "p2p"),
        ("Torrent01.pcap", False, "bittorrent", None, "p2p"),
        ("email2b.pcap", False, "email", None, "email"),
        ("vpn_email2a.pcap", True, "email", None, "email"),
        ("browsing2.pcap", False, "browser", "browsing", None),
    ],
)
def test_iscx_filenames(rules, filename, vpn, application, service, excluded) -> None:
    label = parse_iscx_capture(filename, rules)
    assert label.vpn is vpn
    assert label.application == application
    assert label.service == service
    assert label.excluded_service == excluded


def test_vpn_directory_is_detected_and_nonvpn_is_not(rules) -> None:
    vpn = parse_iscx_capture(Path("data/raw/iscx/VPN-PCAPS-01/youtube1.pcap"), rules)
    clear = parse_iscx_capture(Path("data/raw/iscx/NonVPN-PCAPs-01/youtube1.pcap"), rules)
    assert vpn.vpn is True
    assert clear.vpn is False
    assert vpn.capture_id == "data/raw/iscx/VPN-PCAPS-01/youtube1.pcap"


def test_ambiguous_iscx_names_are_rejected(rules) -> None:
    with pytest.raises(UnmappedLabelError):
        parse_iscx_capture("skype1.pcap", rules)
    with pytest.raises(UnmappedLabelError):
        parse_iscx_capture("facebook1.pcap", rules)


def test_cic_labels_collapse_and_keep_the_binary_view(repo_root: Path) -> None:
    config = load_config(root=repo_root)
    ids2017 = load_security_map(config.labels.cic_ids2017)
    ids2018 = load_security_map(config.labels.cic_ids2018)

    assert map_security_label(" BENIGN ", ids2017) == "benign"
    assert map_security_label("DoS Hulk", ids2017) == "ddos"
    assert map_security_label("FTP-Patator", ids2017) == "brute_force"
    assert map_security_label("PortScan", ids2017) == "port_scan"
    assert map_security_label("Web Attack \u2013 Brute Force", ids2017) == "brute_force"
    assert map_security_label("Web Attack \ufffd XSS", ids2017) == "malware"
    assert map_security_label("Heartbleed", ids2017) == "malware"
    assert set(ids2017.classes) == set(SECURITY_CLASSES)

    assert map_security_label("Infilteration", ids2018) == "malware"
    assert map_security_label("Brute Force -XSS", ids2018) == "malware"
    assert map_security_label("DoS attacks-Hulk", ids2018) == "ddos"
    assert map_security_label("Botnet Ares - Attempted", ids2018) == "botnet"
    assert "port_scan" in ids2018.absent_classes
    assert to_binary_security("ddos") == "malicious"
    assert to_binary_security("benign") == "benign"
    with pytest.raises(UnmappedLabelError):
        map_security_label("NotAnAttack", ids2017)
