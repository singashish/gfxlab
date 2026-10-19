#pragma once

namespace gfx {

class GLPresenter {
public:
    GLPresenter() = default;
    ~GLPresenter();

    GLPresenter(const GLPresenter&) = delete;
    GLPresenter& operator=(const GLPresenter&) = delete;

    bool init();
    bool resize(int width, int height);
    unsigned int texture() const;
    void draw(int width, int height);
    void shutdown();

private:
    unsigned int m_texture = 0;
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    int m_sampler_location = -1;
};

} // namespace gfx
