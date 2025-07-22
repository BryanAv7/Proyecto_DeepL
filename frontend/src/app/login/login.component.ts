import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {

  constructor(private authService: AuthService, private router: Router) {
    // Si el usuario ya está autenticado, redirige al dashboard
    if (this.authService.isLoggedIn()) {
      this.router.navigate(['/dashboard']);
      //La logica ahora se maneja en APP.ROUTES.TS Y Guard
    }
  }

  // Método para iniciar sesión con Google
  onLogin() {
    this.authService.loginWithGoogle();
  }

}
