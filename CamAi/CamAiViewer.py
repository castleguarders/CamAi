import vlc


def playStream(cameraurl):
    vlcInstance = vlc.Instance()
    player = vlcInstance.media_player_new()
    player.set_mrl(cameraurl)
    player.play()
    return player
